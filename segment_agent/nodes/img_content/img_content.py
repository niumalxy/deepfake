from poplib import CR
import re
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from segment_agent.graph.state import AgentState
from segment_agent.nodes.img_content.prompt import get_image_analysis_prompt
from chat_model.openai.langchain_model import model
from entity.segment_agent_status import AgentStatus
from logger import logs
from utils.img_convert import img_to_base64
from segment_agent.graph.state import CroppingImg

def extract_suspicious_regions(state: AgentState, config: Dict[str, Any]):
    """
    从整幅图像中筛选出可疑的不自然区域，供下游检测模块进一步分析。
    
    本函数的角色是"预筛选"——识别图像中所有看起来可疑、不符合自然规律、
    可能存在伪造痕迹的区域，而非对这些区域做出最终的真伪判定。
    
    Args:
        state: AgentState，包含origin_img等信息
        config: 配置字典，包含task_id等配置信息
    
    Returns:
        Dict[str, Any]: 更新后的state字段，包含cropping_imgs（可疑区域列表）
    """
    logs.info("--- Screening Suspicious Regions ---")
    
    origin_img = state.get('origin_img')
    if origin_img is None:
        logs.error("No origin_img found in state")
        return {"status": AgentStatus.FINISHED}
    
    width, height = origin_img.size
    logs.info(f"Image size: {width} x {height}")
    
    img_base64 = img_to_base64(origin_img)
    
    content = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{img_base64}"
        }
    },
    {
        "type": "text",
        "text": f"这是我的待检测图像，图像尺寸：{width} x {height}"
    }]
    
    messages = [
        SystemMessage(content=get_image_analysis_prompt()),
        HumanMessage(content=content)
    ]
    
    response = model.invoke(messages)
    logs.info(f"Response content: {response.content}")
    
    cropping_imgs = []
    
    try:
        json_str = response.content.strip()

        json_str = re.sub(r"<thinking>.*?</thinking>", "", json_str, flags=re.DOTALL | re.IGNORECASE)

        json_match = re.search(r'\{[\s\S]*\}', json_str)
        if json_match:
            json_str = json_match.group()
        
        result = json.loads(json_str)
        
        for part_key, part_value in result.items():
            if part_key.startswith('part_'):
                location = part_value.get('location', (0, 0, 0, 0))
                description = part_value.get('description', '')
                items = part_value.get('items', '')
                anomaly_type = part_value.get('anomaly_type', 'other')

                cropping_imgs.append(CroppingImg(
                    items=items,
                    top_left=location[:2],
                    bottom_right=location[2:],
                    description=description,
                    save_path="",
                    anomaly_type=anomaly_type,
                ))
                logs.info(f"Extracted region: {items}, type: {anomaly_type}, location: {location}, description: {description}")
        
        logs.info(f"Total suspicious regions screened: {len(cropping_imgs)}")
        
    except json.JSONDecodeError as e:
        logs.error(f"Failed to parse JSON response: {e}")
        logs.error(f"Response content: {response.content}")
    except Exception as e:
        logs.error(f"Error processing response: {e}")
    
    return {
        "status": AgentStatus.PLANNING,
        "cropping_imgs": cropping_imgs,
        "content_messages": [response]
    }
