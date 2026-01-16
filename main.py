from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import utils.idgen as idgen
from logger import logs
from init import init
from db.local_map import status_map
import json, os
from entity.agent_status import AgentStatus

app = FastAPI()

# 初始化中间件
init(app)

# 配置模板目录
templates = Jinja2Templates(directory="templates")

@app.get("/")
def render_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/check")
def read_root():
    return {"code": "200", "msg": "success"}

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    return {"status": status_map.get(task_id, AgentStatus.WAITING)}

# 公共函数：读取文件并转换为base64
async def _read_file_to_base64(file: UploadFile):
    content = await file.read()
    import base64
    return base64.b64encode(content).decode('utf-8')

# 公共函数：生成任务ID和记录日志
def _init_task(agent_type: str):
    task_id = idgen.generate_id()
    logs.info(f"Received task {task_id}, starting analysis with {agent_type} agent...")
    status_map[task_id] = AgentStatus.WAITING
    return task_id

# 标准agent处理函数
def _run_standard_agent(img_base64: str, use_chinese: bool, task_id: str):
    from agent.graph import create_graph
    from langchain_core.messages import HumanMessage
    graph = create_graph(task_id=task_id, img=img_base64, use_chinese=use_chinese)
    inputs = {
        "messages": [HumanMessage(content="Start analysis")]
    }
    result = graph.invoke(inputs)
    
    # 处理标准agent的报告
    report = ""
    if "report_messages" in result and result["report_messages"]:
        report = result["report_messages"][-1].content
    
    return report

# segment_agent处理函数
def _run_segment_agent(img_base64: str, task_id: str):
    from segment_agent.graph import create_graph
    from PIL import Image
    import io
    import base64
    
    # 将base64解码为图片对象
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes))
    graph = create_graph(task_id=task_id, img=img)
    inputs = {}
    result = graph.invoke(inputs)
    
    # 处理segment_agent的报告
    report = ""
    if "report" in result:
        report = result["report"]
    elif "final_report" in result:
        report = result["final_report"]
    
    return report

# 标准agent流式处理函数
def _stream_standard_agent(img_base64: str, use_chinese: bool, task_id: str):
    from agent.graph import create_graph
    graph = create_graph(task_id=task_id, img=img_base64, use_chinese=use_chinese)
    inputs = {}
    
    for event in graph.stream(inputs):
        for node, state in event.items():
            status = node
            message = ""
            
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

# segment_agent流式处理函数
def _stream_segment_agent(img_base64: str, task_id: str):
    from segment_agent.graph import create_graph
    from PIL import Image
    import io
    import base64
    
    # 将base64解码为图片对象
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes))
    graph = create_graph(task_id=task_id, img=img)
    inputs = {}
    
    current_cropped_imgs = []
    current_img_idx_val = 0
    
    for event in graph.stream(inputs):
        for node, state in event.items():
            status = node
            message = ""
            
            # Update local tracking
            if "cropped_imgs" in state:
                current_cropped_imgs = state["cropped_imgs"]
            if "current_img_idx" in state:
                current_img_idx_val = state["current_img_idx"]
            
            # segment_agent的状态处理
            if node == "img_content":
                message = "Extracting image content and identifying suspicious regions"
                # Show inference process (model reasoning)
                if "content_messages" in state and state["content_messages"]:
                    last_msg = state["content_messages"][-1]
                    if hasattr(last_msg, 'content'):
                        content = last_msg.content
                        if isinstance(content, str):
                            # Use small text for inference process
                            message = f'<div style="font-size: 0.8em; color: #666;">{content}</div>'
                yield json.dumps({
                        "current_node": "内容分析",
                    }, ensure_ascii=False) + "\n"

            elif node == "img_cropping":
                message = "Segmenting image into parts for detailed analysis"
                # 添加cropped_imgs到流式输出
                if current_cropped_imgs:
                    yield json.dumps({
                        "current_node": "局部提取 ()",
                        "status": status, 
                        "message": message,
                        "cropped_imgs": current_cropped_imgs,
                        "current_task": 0,
                        "total_tasks": len(current_cropped_imgs)
                    }, ensure_ascii=False) + "\n"
                    continue
            elif node == "img_part_analysis":
                message = "Analyzing image part for deepfake indicators"
                
                # Try to get the actual analysis content
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
                
                # Wrap in small text for inference process
                if message and message != "Analyzing image part for deepfake indicators":
                    message = f'<div style="font-size: 0.8em; color: #666;">{message}</div>'
                
                # 获取当前分析索引
                total_imgs = len(current_cropped_imgs)
                # 更新cropped_imgs状态
                if current_cropped_imgs:
                    yield json.dumps({
                        "current_node": "图像区域分析",
                        "status": status, 
                        "message": message,
                        "cropped_imgs": current_cropped_imgs,
                        "current_task": current_img_idx_val + 1,
                        "total_tasks": total_imgs
                    }, ensure_ascii=False)
                    continue
            elif node == "tool_call":
                continue

            elif node == "next_part":
                # Yield task completion update
                # current_img_idx_val is already updated to the next index
                finished_idx = current_img_idx_val - 1
                result_text = ""
                if 0 <= finished_idx < len(current_cropped_imgs):
                    is_done = current_cropped_imgs[finished_idx]["is_done"]
                    result_text = current_cropped_imgs[finished_idx]["analysis_result"]
                    if is_done:
                        message = f"Part {finished_idx + 1} analysis completed: {result_text}"
                else:
                    message = f"Part {finished_idx + 1} analysis failed"
                
                yield json.dumps({
                    "current_node": "图像区域分析",
                    "status": "task_completed", 
                    "result": result_text,
                    "cropped_imgs": current_cropped_imgs,
                    "current_task": current_img_idx_val,
                    "total_tasks": len(current_cropped_imgs)
                }, ensure_ascii=False) + "\n"
                
            elif node == "report":
                status = "summary" # Route to Analysis Report section
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
            
            total_tasks = len(state.get("tasks", [])) if "tasks" in state else 0
            current_task = state.get("current_task", 0)
            
            yield json.dumps({
                "status": status, 
                "message": message,
                "current_task": current_task,
                "total_tasks": total_tasks
            }, ensure_ascii=False) + "\n"
    
    yield json.dumps({"status": "finished", "message": "Analysis complete"}, ensure_ascii=False) + "\n"

# 标准agent API端点
@app.post("/api/task/standard")
async def analyze_task_standard(file: UploadFile = File(...), use_chinese: bool = False):
    from fastapi.concurrency import run_in_threadpool
    
    # 读取图片并转换为base64
    img_base64 = await _read_file_to_base64(file)
    
    # 初始化任务
    task_id = _init_task("standard")

    try:
        report = await run_in_threadpool(_run_standard_agent, img_base64, use_chinese, task_id)
        
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

# segment_agent API端点
@app.post("/api/task/segment")
async def analyze_task_segment(file: UploadFile = File(...)):
    from fastapi.concurrency import run_in_threadpool
    
    # 读取图片并转换为base64
    img_base64 = await _read_file_to_base64(file)
    
    # 初始化任务
    task_id = _init_task("segment")

    try:
        report = await run_in_threadpool(_run_segment_agent, img_base64, task_id)
        
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

# 标准agent流式API端点
@app.post("/api/task/stream/standard")
async def analyze_task_stream_standard(file: UploadFile = File(...), use_chinese: bool = False):
    # 读取图片并转换为base64
    img_base64 = await _read_file_to_base64(file)
    
    # 初始化任务
    task_id = _init_task("standard")

    def stream_generator():
        try:
            yield from _stream_standard_agent(img_base64, use_chinese, task_id)
        except Exception as e:
            logs.error(f"Task {task_id} failed: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

# segment_agent流式API端点
@app.post("/api/task/stream/segment")
async def analyze_task_stream_segment(file: UploadFile = File(...)):
    # 读取图片并转换为base64
    img_base64 = await _read_file_to_base64(file)
    
    # 初始化任务
    task_id = _init_task("segment")

    def stream_generator():
        try:
            yield from _stream_segment_agent(img_base64, task_id)
        except Exception as e:
            logs.error(f"Task {task_id} failed: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    # 检查必要文件夹是否存在
    def check_necessary_dir(path):
        if not os.path.exists(path):
            logs.info(f"Create necessary directory: {path}")
            os.makedirs(path)
    check_necessary_dir("agent/plan/docs")
    check_necessary_dir("agent/summary/docs")
    # 为segment_agent创建必要的文件夹
    check_necessary_dir("segment_agent/plan/docs")
    check_necessary_dir("segment_agent/summary/docs")

    import uvicorn
    logs.info("Start deepfakeagentdemo! You can access it at http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000)
    
