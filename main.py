from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import utils.idgen as idgen
from logger import logs
from init import init
from db.local_map import status_map
from entity.agent_status import AgentStatus
import json, os

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

@app.post("/api/task/stream")
async def analyze_task_stream(file: UploadFile = File(...)):
    import base64
    from agent.graph import create_graph
    from langchain_core.messages import HumanMessage
    
    # 读取图片并转换为base64
    content = await file.read()
    img_base64 = base64.b64encode(content).decode('utf-8')
    
    task_id = idgen.generate_id()
    logs.info(f"Received task {task_id}, starting streaming analysis...")
    status_map[task_id] = AgentStatus.WAITING

    def stream_generator():
        # try:
            graph = create_graph(task_id=task_id, img=img_base64)
            inputs = {
                "messages": [HumanMessage(content="Start analysis")]
            }
            
            for event in graph.stream(inputs):
                for node, state in event.items():
                    status = node
                    message = ""
                    
                    if node == "img_content":
                        message = "Image content extracted"
                    elif node == "plan":
                        message = state.get("plan", "")
                    elif node == "tasks":
                        tasks = state.get("tasks", [])
                        message = f"Generated {len(tasks)} tasks"
                    elif node == "analysis":
                        msgs = state.get("analysis_messages", [])
                        if msgs:
                            message = msgs[-1].content
                    elif node == "summary":
                        msgs = state.get("report_messages", [])
                        if msgs:
                            message = msgs[-1].content
                    
                    yield json.dumps({"status": status, "message": message}, ensure_ascii=False) + "\n"
            
            yield json.dumps({"status": "finished", "message": "Analysis complete"}, ensure_ascii=False) + "\n"
            
        # except Exception as e:
        #     logs.error(f"Task {task_id} failed: {str(e)}")
        #     yield json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    # 检查必要文件夹是否存在
    def check_necessary_dir(path):
        if not os.path.exists(path):
            logs.info(f"Create necessary directory: {path}")
            os.makedirs(path)
    check_necessary_dir("agent/plan/docs")
    check_necessary_dir("agent/summary/docs")

    import uvicorn
    logs.info("Start deepfakeagentdemo! You can access it at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
