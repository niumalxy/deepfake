from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from segment_agent.graph.state import AgentState
from entity.segment_agent_status import AgentStatus
from logger import logs
from chat_model.openai.langchain_model import model
from utils.img_convert import img_to_base64
from segment_agent.nodes.report.prompt import get_summary_system_prompt


def report(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成最终的分析报告节点
    
    Args:
        state: AgentState，包含当前工作流状态
        config: 配置字典
    
    Returns:
        Dict[str, Any]: 更新后的状态字段
    """
    logs.info("--- Generating Final Report with Model ---")
    
    # 获取分析消息
    analysis_messages = state.get('cropped_imgs', [])

    # 获取原始图像并转换为base64
    origin_img = state.get('origin_img')
    if origin_img:
        img_base64 = img_to_base64(origin_img)
        origin_img_content = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        }]
    else:
        origin_img_content = []
        logs.warning("No origin image found for report generation")
    
    # 提取分析结果文本
    analysis_text = "以下是对图片局部的详细分析结果："
    for idx, msg in enumerate(analysis_messages):
        this_analysis = {
            "location": msg["items"],
            "analysis_result": msg["analysis_result"],
        }
        analysis_text += f"\n{this_analysis}"
    
    # 准备用户消息内容
    user_content = []
    
    # 添加原始图像（如果存在）
    if origin_img_content:
        user_content.extend(origin_img_content)
    
    # 添加分析结果文本
    user_content.append({
        "type": "text",
        "text": f"以下是图像各部分的详细分析结果，请基于这些信息生成一份完整的分析报告：\n\n{analysis_text}"
    })
    
    # 获取系统提示词（根据配置确定是否使用中文）
    use_chinese = config.get('use_chinese', False)
    system_prompt = get_summary_system_prompt(use_chinese)
    
    # 准备消息列表
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    # 调用模型生成报告
    try:
        response = model.invoke(messages)
        generated_report = response.content
        
        logs.info("Final report generated successfully with model")
        
        return {
            "report": generated_report,
            "status": AgentStatus.FINISHED,
        }
    except Exception as e:
        logs.error(f"Error generating report with model: {e}")
        
        # 如果模型调用失败，返回基本报告
        return {
            "report": "",
            "status": AgentStatus.FINISHED,
        }