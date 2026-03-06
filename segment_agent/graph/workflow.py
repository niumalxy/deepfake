import functools
from PIL import Image
from typing import Optional
from segment_agent.nodes.img_content.img_content import extract_suspicious_regions
from segment_agent.nodes.img_segment.img_segment import crop_image_by_coords
from segment_agent.nodes.img_part_analysis.img_part_analysis import analyze_partial_image
from segment_agent.nodes.next_part.next_part import next_part
from segment_agent.nodes.tool_call.tool_call import tool_call
from segment_agent.nodes.report.report import report
from segment_agent.nodes.dump2db.save import dump2db
from entity.segment_agent_status import AgentStatus
from segment_agent.graph.state import AgentState
from langgraph.graph import StateGraph, START, END
from entity.segment_agent_config import SegmentAgentConfig


def should_continue_analysis(state: AgentState) -> str:
    last_message = state["analysis_messages"][-1]
    # 检查最后一条消息是否包含工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_call"
    content = getattr(last_message, 'content', '')
    if "<complete>" in content:
        return "complete"
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


def have_next_part(state: AgentState) -> str:
    """
    检查是否还有下一个分析部分
    
    Args:
        state: AgentState
    
    Returns:
        str: "has_regions" 或 "no_regions"
    """
    current_idx = state.get('current_img_idx', 0)
    cropped_imgs = state.get('cropped_imgs', [])
    if current_idx == len(cropped_imgs):
        return "no"
    return "yes"

def create_graph(task_id: str, img: Image.Image, use_chinese: bool = True, label: Optional[tuple[int, str]] = None): # label取值为tuple, (int, str), 代表(类别, 描述)
    """
    创建segment_agent的工作流图
    
    Args:
        task_id: 任务ID
        img: 原始图像
        use_chinese: 是否使用中文
    
    Returns:
        编译后的工作流图
    """
    config = SegmentAgentConfig(task_id=task_id, use_chinese=use_chinese, label=label)
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
    workflow.add_node("tool_call", functools.partial(tool_call, config=config))
    workflow.add_node("next_part", next_part)
    workflow.add_node("report", functools.partial(report, config=config))
    workflow.add_node("workflow_end", functools.partial(dump2db, config=config))
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
            "no_regions": "workflow_end"
        }
    )
    
    workflow.add_edge("img_cropping", "img_part_analysis")
    workflow.add_edge("tool_call", "img_part_analysis")
    # 决定是否继续分析
    workflow.add_conditional_edges(
        "img_part_analysis",
        should_continue_analysis,
        {
            "tool_call": "tool_call",
            "continue": "img_part_analysis",
            "complete": "next_part"
        }
    )
    workflow.add_conditional_edges("next_part", have_next_part, {
        "yes": "img_part_analysis",
        "no": "report"
    })
    workflow.add_edge("report", "workflow_end")

    return workflow.compile()
