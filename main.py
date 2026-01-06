from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
import utils.idgen as idgen
from logger import logs
from init import init
from db.local_map import status_map
from entity.agent_status import AgentStatus

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

@app.post("/api/task")
async def analyze_task(file: UploadFile = File(...)):
    import base64
    import uuid
    from agent.graph import create_graph
    from langchain_core.messages import HumanMessage
    from fastapi.concurrency import run_in_threadpool
    
    # 读取图片并转换为base64
    content = await file.read()
    img_base64 = base64.b64encode(content).decode('utf-8')
    
    task_id = idgen.generate_id()
    logs.info(f"Received task {task_id}, starting analysis...")

    status_map[task_id] = AgentStatus.WAITING

    def _run_agent():
        graph = create_graph(task_id=task_id, img=img_base64)
        inputs = {
            "messages": [HumanMessage(content="Start analysis")]
        }
        return graph.invoke(inputs)

    try:
        result = await run_in_threadpool(_run_agent)
        
        report = ""
        if "report_messages" in result and result["report_messages"]:
            report = result["report_messages"][-1].content
            
        return {
            "code": 200, 
            "msg": "success", 
            "data": {
                "task_id": task_id,
                "report": report
            }
        }
    except Exception as e:
        logs.error(f"Task {task_id} failed: {str(e)}")
        return {"code": 500, "msg": str(e)}

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    return {"status": status_map.get(task_id, AgentStatus.WAITING)}

if __name__ == "__main__":
    import uvicorn
    logs.info("Start deepfakeagentdemo! You can access it at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
