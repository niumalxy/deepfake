from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from segment_agent.graph.state import AgentState, CroppedImg
from segment_agent.nodes.img_part_analysis.prompt import get_partial_image_analysis_prompt
from chat_model.openai.langchain_model import model
from entity.segment_agent_status import AgentStatus
from logger import logs
from utils.img_convert import img_path_to_base64, img_to_base64
from segment_agent.skills.tools.registry import TOOLS_SCHEMA


def analyze_partial_image(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    逐一对分割的图像部分进行深度伪造检测
    
    Args:
        state: AgentState，包含cropped_imgs等信息
        config: 配置字典，包含task_id等配置信息
    
    Returns:
        Dict[str, Any]: 更新后的state字段
    """
    logs.info("--- Analyzing Partial Image ---")
    
    cropped_imgs = state.get('cropped_imgs', [])
    if not cropped_imgs:
        logs.warning("No cropped images to analyze")
        return {"status": AgentStatus.FINISHED}
    
    current_idx = state.get('current_img_idx', 0)
    current_img = cropped_imgs[current_idx]
    logs.info(f"Analyzing image {current_idx + 1}/{len(cropped_imgs)}: {current_img['save_path']}")
    try:
        img_base64 = img_path_to_base64(current_img['save_path'])
        desc = {
            "items": current_img['items'],
            "description": current_img['description'],
        }
        content = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        },
        {
            "type": "text",
            "text": f"以下为当前区域的简要描述，你既可以选择参考，也可以选择不参考，注意不要过分关注：\n{desc}"
        }]
        origin_img_content = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_to_base64(state['origin_img'])}"
            }
        }]
        prompt = get_partial_image_analysis_prompt()

        # 历史记录
        analysis_messages = state.get('analysis_messages', [])
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=content),
            HumanMessage(content=origin_img_content)
        ]  if not analysis_messages else analysis_messages
        
        # 绑定工具到模型，使模型知道可以使用哪些工具
        bound_model = model.bind_tools(TOOLS_SCHEMA)
        response = bound_model.invoke(messages)
        
        return {
            "analysis_messages": messages + [response],
        }
        
    except Exception as e:
        logs.error(f"Error analyzing image {current_idx + 1}: {e}")
        cropped_imgs[current_idx]['is_done'] = True
        cropped_imgs[current_idx]['analysis_result'] = f"Error: {str(e)}"
        
        next_idx = current_idx + 1
        if next_idx >= len(cropped_imgs):
            status = AgentStatus.FINISHED
        else:
            status = AgentStatus.ANALYZING
        
        return {
            "status": status,
            "cropped_imgs": cropped_imgs,
            "current_analysis_idx": next_idx
        }
