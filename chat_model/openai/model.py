import openai
from chat_model.conf.conf import get_model_conf

class ChatModel:
    def __init__(self):
        model_conf = get_model_conf()
        openai.api_key = model_conf["api_key"]
        openai.api_base = model_conf["api_base"]
        self.model_name = model_conf["model_name"]
        self.chat_model = openai.ChatCompletion

    def generate(self, messages: list) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )
        return response.choices[0].message.content