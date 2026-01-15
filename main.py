from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import utils.idgen as idgen
from logger import logs
from init import init
from db.local_map import status_map
import json, os
from enum import Enum
from entity.agent_status import AgentStatus

app = FastAPI()

# 初始化中间件
init(app)

# 配置模板目录
templates = Jinja2Templates(directory="templates")

# 定义可用的agent类型
class AgentType(str, Enum):
    STANDARD_AGENT = "standard"
    SEGMENT_AGENT = "segment"

@app.get("/")
def render_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/check")
def read_root():
    return {"code": "200", "msg": "success"}

@app.post("/api/task")
async def analyze_task(file: UploadFile = File(...), agent_type: AgentType = AgentType.STANDARD_AGENT, use_chinese: bool = False):
    import base64
    import uuid
    from langchain_core.messages import HumanMessage
    from fastapi.concurrency import run_in_threadpool
    
    # 读取图片并转换为base64
    content = await file.read()
    img_base64 = base64.b64encode(content).decode('utf-8')
    
    task_id = idgen.generate_id()
    logs.info(f"Received task {task_id}, starting analysis with {agent_type} agent...")

    status_map[task_id] = AgentStatus.WAITING

    def _run_agent():
        if agent_type == AgentType.SEGMENT_AGENT:
            # 使用 segment_agent
            from segment_agent.graph import create_graph
            from PIL import Image
            import io
            # 将base64解码为图片对象
            import base64
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            graph = create_graph(task_id=task_id, img=img)
            inputs = {"status": AgentStatus.INITIATING}
        else:
            # 使用标准agent
            from agent.graph import create_graph
            graph = create_graph(task_id=task_id, img=img_base64, use_chinese=use_chinese)
            inputs = {
                "messages": [HumanMessage(content="Start analysis")]
            }
        return graph.invoke(inputs)

    try:
        result = await run_in_threadpool(_run_agent)
        
        report = ""
        if agent_type == AgentType.SEGMENT_AGENT:
            # segment_agent的报告结构
            if "final_report" in result:
                report = result["final_report"]
        else:
            # standard agent的报告结构
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
async def analyze_task_stream(file: UploadFile = File(...), agent_type: AgentType = AgentType.STANDARD_AGENT, use_chinese: bool = False):
    import base64
    from langchain_core.messages import HumanMessage
    
    # 读取图片并转换为base64
    content = await file.read()
    img_base64 = base64.b64encode(content).decode('utf-8')
    
    task_id = idgen.generate_id()
    logs.info(f"Received task {task_id}, starting streaming analysis with {agent_type} agent...")
    status_map[task_id] = AgentStatus.WAITING

    def stream_generator():
        #try:
            if agent_type == AgentType.SEGMENT_AGENT:
                # 使用 segment_agent
                from segment_agent.graph import create_graph
                from PIL import Image
                import io
                # 将base64解码为图片对象
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(io.BytesIO(img_bytes))
                graph = create_graph(task_id=task_id, img=img)
            else:
                # 使用标准agent
                from agent.graph import create_graph
                graph = create_graph(task_id=task_id, img=img_base64, use_chinese=use_chinese)
            inputs = {}
            for event in graph.stream(inputs):
                for node, state in event.items():
                    status = node
                    message = ""
                    
                    if agent_type == AgentType.SEGMENT_AGENT:
                        # segment_agent的状态处理
                        if node == "img_content":
                            message = "Extracting image content and identifying suspicious regions"
                        elif node == "img_cropping":
                            message = "Segmenting image into parts for detailed analysis"
                        elif node == "img_part_analysis":
                            message = "Analyzing image part for deepfake indicators"
                        elif node == "tool_call":
                            message = "Executing tools for enhanced analysis"
                        elif node == "next_part":
                            message = "Determining next region to analyze"
                        elif node == "report":
                            message = "Generating comprehensive deepfake detection report"
                            # Get report content
                            if "report" in state:
                                message = state["report"]
                        else:
                            # General handling for other nodes
                            if "analysis_messages" in state and state["analysis_messages"]:
                                msgs = state["analysis_messages"]
                                if msgs:
                                    last_msg = msgs[-1]
                                    if hasattr(last_msg, 'content'):
                                        if isinstance(last_msg.content, str):
                                            message = last_msg.content
                                        elif isinstance(last_msg.content, list):
                                            for item in last_msg.content:
                                                if isinstance(item, dict) and 'text' in item:
                                                    message = item['text']
                                                    break
                                                
                    else:
                        # standard agent的状态处理
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
                    
                    total_tasks = len(state.get("tasks", [])) if "tasks" in state else 0
                    current_task = state.get("current_task", 0)
                    
                    yield json.dumps({
                        "status": status, 
                        "message": message,
                        "current_task": current_task,
                        "total_tasks": total_tasks
                    }, ensure_ascii=False) + "\n"
            
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
    
