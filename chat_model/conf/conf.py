import yaml

def get_model_conf():
    # 读取配置
    with open("chat_model/conf/conf.yml", "r") as f:
        model_conf = yaml.safe_load(f)["qwen-vl"]
    return model_conf
