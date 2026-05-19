from langchain_openai import ChatOpenAI
from chat_model.conf.conf import get_model_conf
import os

class ChatModel(ChatOpenAI):
    def __init__(self, **kwargs):
        model_conf = get_model_conf()
        
        # Mapping config keys to ChatOpenAI parameters
        # ChatOpenAI keys: api_key, base_url, model, max_tokens, etc.
        
        defaults = {
            "api_key": model_conf.get("api_key"),
            "base_url": model_conf.get("api_base"),
            "model": model_conf.get("model") or model_conf.get("model_name"),
            "max_tokens": model_conf.get("max_tokens"),
        }
        # 仅当 conf 中显式配置 user_agent 时才覆盖 (用于绕过中转网关 UA 拦截)
        if model_conf.get("user_agent"):
            defaults["default_headers"] = {"User-Agent": model_conf["user_agent"]}
        
        # 收集额外的模型参数（如 enable_thinking）通过 extra_body 传递
        extra_body = {}
        if "enable_thinking" in model_conf:
            extra_body["enable_thinking"] = model_conf["enable_thinking"]
        if extra_body:
            defaults["model_kwargs"] = {"extra_body": extra_body}
        
        # Update defaults with any provided kwargs (kwargs take precedence)
        for key, value in defaults.items():
            if key not in kwargs and value is not None:
                kwargs[key] = value
                
        # Handle cases where keys might be passed as openai_api_key, etc.
        if "openai_api_key" not in kwargs and "api_key" in kwargs:
            kwargs["openai_api_key"] = kwargs.pop("api_key")
        if "openai_api_base" not in kwargs and "base_url" in kwargs:
            kwargs["openai_api_base"] = kwargs.pop("base_url")

        super().__init__(**kwargs)

model = ChatModel()