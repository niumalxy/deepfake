import functools
from PIL import Image
from segment_agent.nodes.img_content.img_content import extract_suspicious_regions
from segment_agent.nodes.img_segment.img_segment import crop_image_by_coords
from segment_agent.nodes.img_part_analysis.img_part_analysis import analyze_partial_image
from entity.segment_agent_status import AgentStatus
from segment_agent.graph.state import AgentState
from langgraph.graph import StateGraph, START, END
from entity.segment_agent_config import SegmentAgentConfig


def should_continue_analysis(state: AgentState) -> str:
    """
    决定是否继续分析下一个图像部分
    
    Args:
        state: AgentState
    
    Returns:
        str: "continue" 或 "end"
    """
    status = state.get('status')
    if status == AgentStatus.FINISHED:
        return "end"
    return "continue"


def has_suspicious_regions(state: AgentState) -> str:
    """
    检查是否有可疑区域需要分析
    
    Args:
        state: AgentState
    
    Returns:
        str: "has_regions" 或 "no_regions"
    """
    cropping_imgs = state.get('cropping_imgs', [])
    if cropping_imgs:
        return "has_regions"
    return "no_regions"


def create_graph(task_id: str, img: Image.Image):
    """
    创建segment_agent的工作流图
    
    Args:
        task_id: 任务ID
        img: 原始图像
    
    Returns:
        编译后的工作流图
    """
    config = SegmentAgentConfig(task_id=task_id)
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("init", lambda state: {
        "status": AgentStatus.INITIATING, 
        "origin_img": img,
        "cropping_imgs": [],
        "cropped_imgs": [],
        "current_analysis_idx": 0
    })
    workflow.add_node("img_content", functools.partial(extract_suspicious_regions, config=config))
    workflow.add_node("img_cropping", functools.partial(crop_image_by_coords, config=config))
    workflow.add_node("img_part_analysis", functools.partial(analyze_partial_image, config=config))

    # 设置入口点
    workflow.set_entry_point("init")

    # 添加边
    workflow.add_edge("init", "img_content")
    
    # 检查是否有可疑区域
    workflow.add_conditional_edges(
        "img_content",
        has_suspicious_regions,
        {
            "has_regions": "img_cropping",
            "no_regions": END
        }
    )
    
    workflow.add_edge("img_cropping", "img_part_analysis")
    
    # 决定是否继续分析
    workflow.add_conditional_edges(
        "img_part_analysis",
        should_continue_analysis,
        {
            "continue": "img_part_analysis",
            "end": END
        }
    )

    return workflow.compile()
