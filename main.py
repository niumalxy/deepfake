from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from logger import logs
from init import init

app = FastAPI()

# 初始化中间件
init(app)

# 配置模板目录
templates = Jinja2Templates(directory="templates")

@app.get("/")
def render_index():
    return templates.TemplateResponse("index.html", {"request": Request})

@app.get("/check")
def read_root():
    return {"code": "200", "msg": "success"}


if __name__ == "__main__":
    import uvicorn
    logs.info("Start deepfakeagentdemo! You can access it at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
