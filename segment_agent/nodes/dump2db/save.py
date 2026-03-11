from weakref import ref
from entity.segment_agent_config import SegmentAgentConfig
from db.mongodb import insert_analysis, insert_segment
from segment_agent.graph.state import AgentState
from utils.img_convert import img_to_base64
from mq.reflection_produce import reflection_produce
from entity.dump_type import DumpType
from entity.segment_agent_status import AgentStatus
from utils.img_embedding import img_embedder
from segment_agent.rag.faiss_db import faiss_manager
import asyncio
from logger import logs

# 保存分析结果到数据库
def dump2db(state: AgentState, config: SegmentAgentConfig) -> None:
    """
    将分析结果保存到数据库
    
    Args:
        state: 工作流状态
        config: 配置参数
    """
    # 如果没有可疑区域，需要补充normal标签
    if not state["cropped_imgs"]:
        state["prediction"] = "normal"

    # 总是将图像向量插入 FAISS 用于 RAG
    if state.get("origin_img"):
        try:
            emb = img_embedder.get_embedding(state["origin_img"])
            faiss_manager.insert_vector(emb, config["task_id"])
        except Exception as e:
            logs.error(f"Failed to insert image embedding to FAISS for task {config['task_id']}: {e}")

    # 如果结果正确，或状态为INVALID，暂时不分析
    if config["label"] is None or ((state["prediction"] == "fake") ^ (config["label"][0] == 0)) or state["status"] == AgentStatus.INVALID:
        return

    async def dump_analysis():
        analysis = {
            "task_id": config["task_id"],
            "origin_img": img_to_base64(state["origin_img"]),
            "label": config["label"][0],
            "description": config["label"][1],
            "prediction": state["prediction"],
            "report": state["report"]
        }
        # 如果结果不对，需要分析，先存入MongoDB，再在MQ中生产该记录
        insert_analysis(analysis)

        # 先做处理
        segment_images = []
        for item in state["cropped_imgs"]:
            segment_images.append({
                "top_left": item["top_left"],
                "bottom_right": item["bottom_right"],
                "description": item["description"],
            })
        segment = {
            "task_id": config["task_id"],
            "origin_img": img_to_base64(state["origin_img"]),
            "segment_images": segment_images,
        }
        # 保存分割结果
        insert_segment(segment)

        # MQ生产消息
        reflection_produce({"task_id": config["task_id"]}, DumpType.SEGMENT)
        reflection_produce({"task_id": config["task_id"]}, DumpType.REPORT)
    
    asyncio.run(dump_analysis())

