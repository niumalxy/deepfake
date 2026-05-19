from weakref import ref
from entity.segment_agent_config import SegmentAgentConfig
from db.mongodb import insert_analysis, insert_segment
from segment_agent.graph.state import AgentState
from utils.img_convert import img_to_base64
from mq.reflection_produce import reflection_produce
from entity.dump_type import DumpType
from entity.segment_agent_status import AgentStatus
from segment_agent.rag.faiss_db import faiss_manager
import asyncio
from logger import logs

def _get_img_embedder():
    from utils.img_embedding import img_embedder
    return img_embedder

# 保存分析结果到数据库
def dump2db(state: AgentState, config: SegmentAgentConfig):
    """
    将分析结果保存到数据库

    Args:
        state: 工作流状态
        config: 配置参数

    Returns:
        Dict[str, Any]: 状态更新（至少包含 prediction）。LangGraph 不会捕获节点内对 state 的就地修改，
        只有返回的 dict 才会被合并进 graph state。
    """
    state_updates = {}

    # 如果没有可疑区域，需要补充normal标签
    if not state["cropped_imgs"]:
        state["prediction"] = "normal"
        state_updates["prediction"] = "normal"

    if config.get("skip_dump"):
        logs.info(f"skip_dump=True for task {config['task_id']}, skipping embedding + MongoDB + MQ dump.")
        return state_updates

    # 将图像向量插入 FAISS 用于 RAG（仅在非 skip_dump 模式下）
    if state.get("origin_img"):
        try:
            emb = _get_img_embedder().get_embedding(state["origin_img"])
            faiss_manager.insert_vector(emb, config["task_id"])
        except Exception as e:
            logs.error(f"Failed to insert image embedding to FAISS for task {config['task_id']}: {e}")

    # 如果结果正确，或状态为INVALID，暂时不分析
    if config["label"] is None or ((state["prediction"] == "fake") ^ (config["label"][0] == 0)) or state["status"] == AgentStatus.INVALID:
        return state_updates

    async def dump_analysis():
        cropping_imgs = state.get("cropping_imgs", []) or []
        cropped_imgs = state.get("cropped_imgs", []) or []
        anomaly_types = sorted({c.get("anomaly_type", "other") for c in cropping_imgs}) if cropping_imgs else []
        n_screened = len(cropping_imgs) or len(cropped_imgs)

        analysis = {
            "task_id": config["task_id"],
            "origin_img": img_to_base64(state["origin_img"]),
            "label": config["label"][0],
            "description": config["label"][1],
            "prediction": state["prediction"],
            "report": state["report"],
            # Phase 4 routing signals consumed by reflect_prompt()
            "n_screened": n_screened,
            "anomaly_types": ",".join(anomaly_types) if anomaly_types else "none",
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
    return state_updates

