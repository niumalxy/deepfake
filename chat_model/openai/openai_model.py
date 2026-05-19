import json
from openai import OpenAI
from typing import List, Optional, Any, Union
from chat_model.conf.conf import get_model_conf
from agent.tools.registry import TOOLS_SCHEMA, TOOLS_MAPPING

class ChatModel:
    def __init__(self):
        model_conf = get_model_conf()
        client_kwargs = {
            "api_key": model_conf["api_key"],
            "base_url": model_conf["api_base"],
        }
        # 仅当 conf 中显式配置 user_agent 时才覆盖 (用于绕过中转网关 UA 拦截)
        if model_conf.get("user_agent"):
            client_kwargs["default_headers"] = {"User-Agent": model_conf["user_agent"]}
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_conf.get("model", model_conf.get("model_name"))
        self.max_tokens = model_conf["max_tokens"]
        self.tools = TOOLS_SCHEMA
        self.tools_mapping = TOOLS_MAPPING
        self.extra_body = {}
        if "enable_thinking" in model_conf:
            self.extra_body["enable_thinking"] = model_conf["enable_thinking"]

    def generate(self, messages: list, tools: Optional[List] = None, tool_choice: Optional[str] = None) -> Union[str, Any]:
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            params["extra_body"] = self.extra_body
        
        current_tools = tools if tools is not None else self.tools
        if current_tools:
            params["tools"] = current_tools
            if tool_choice:
                params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        message = response.choices[0].message
        
        # Tool execution loop
        while message.tool_calls:
            print(f"[DEBUG] Processing tool calls: {message.tool_calls}")
            messages.append(message)
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name in self.tools_mapping:
                    function_to_call = self.tools_mapping[function_name]
                    try:
                        function_response = function_to_call(**function_args)
                        # Ensure response is a string
                        if not isinstance(function_response, str):
                            function_response = str(function_response)
                    except Exception as e:
                        function_response = f"Error executing tool {function_name}: {str(e)}"
                else:
                    function_response = f"Error: Tool {function_name} not found"
                
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            
            # Call model again with tool outputs
            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message

        return message.content


if __name__ == "__main__":
    # test generate
    chat_model = ChatModel()
    messages = [{"role": "user", "content": "Call test_tool."}]
    print("--- Attempting to force tool call ---")
    # Force tool choice
    tool_choice = {"type": "function", "function": {"name": "test_tool"}}
    messages = [{"role": "user", "content": "Call test_tool with dummy='hello'."}]
    try:
        response = chat_model.generate(messages, tool_choice=tool_choice)
        print("Final Response:", response)
    except Exception as e:
        print(f"Error with forced tool choice: {e}")
        print("--- Retrying with tool_choice='auto' ---")
        response = chat_model.generate(messages, tool_choice="auto")
        print("Final Response:", response)