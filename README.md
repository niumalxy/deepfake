### 项目介绍
  该项目是一个基于Deepfake技术的智能代理系统，旨在帮助用户识别Deepfake图像。

### Quick Start
#### 安装uv
```bash
pip install uv
```
#### 安装依赖
```bash
uv venv
source .venv/bin/activate
uv sync
```

#### conf
#####需要增加的配置项
- openai api key
    路径： chat_model/conf/conf.yaml
    格式：
    ```yaml
    qwen-vl:
        api_key: "sk-xxxx"
        api_base: "https://api.openai.com/v1"
        model: "qwen-vl"
        max_tokens: 2048
    ```
- db config
    路径： db/conf/conf.yaml
    格式：
    ```yaml
    db:
        host: "localhost"
        port: 3306
        user: "root"
        password: "123456"
        database: "deepfake"
    ```
- 其他配置项

### Agent流程图
    ![Agent流程图](agent_flow.png)