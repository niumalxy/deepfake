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