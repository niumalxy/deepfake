import yaml
from openai import OpenAI

CONF_PATH = "chat_model/conf/conf.yml"
MODEL_KEY = "gpt-5.4"

with open(CONF_PATH, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)[MODEL_KEY]

print(f"[INFO] Using model conf: {MODEL_KEY}")
print(f"[INFO] api_base = {conf['api_base']}")
print(f"[INFO] model    = {conf['model']}")

client_kwargs = {
    "api_key": conf["api_key"],
    "base_url": conf["api_base"],
}
if conf.get("user_agent"):
    client_kwargs["default_headers"] = {"User-Agent": conf["user_agent"]}

client = OpenAI(**client_kwargs)

response = client.chat.completions.create(
    model=conf["model"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, please introduce yourself in one sentence."},
    ],
    stream=False,
)

print("\n[RESPONSE]")
print(response.choices[0].message.content)
