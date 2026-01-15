from poplib import CR
import re
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from segment_agent.graph.state import AgentState
from segment_agent.nodes.img_content.prompt import IMAGE_ANALYSIS_PROMPT
from chat_model.openai.langchain_model import model
from entity.segment_agent_status import AgentStatus
from logger import logs
from utils.img_convert import img_to_base64
from segment_agent.graph.state import CroppingImg

def extract_suspicious_regions(state: AgentState, config: Dict[str, Any]):
    """
    提取图像中可能为伪造的区域
    
    Args:
        state: AgentState，包含origin_img等信息
        config: 配置字典，包含task_id等配置信息
    
    Returns:
        Dict[str, Any]: 更新后的state字段
    """
    logs.info("--- Extracting Suspicious Regions ---")
    
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
        SystemMessage(content=IMAGE_ANALYSIS_PROMPT),
        HumanMessage(content=content)
    ]
    
    response = model.invoke(messages)
    logs.info(f"Response content: {response.content}")
    
    cropping_imgs = []
    
    try:
        json_str = response.content.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', json_str)
        if json_match:
            json_str = json_match.group()
        
        result = json.loads(json_str)
        
        for part_key, part_value in result.items():
            if part_key.startswith('part_'):
                location = part_value.get('location', '')
                coordinates_str = part_value.get('coordinates', '')
                description = part_value.get('description', '')
                
                coords_match = re.search(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', coordinates_str)
                if coords_match:
                    x1 = int(coords_match.group(1))
                    y1 = int(coords_match.group(2))
                    x2 = int(coords_match.group(3))
                    y2 = int(coords_match.group(4))
                    
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    if x2 > x1 and y2 > y1:
                        cropping_imgs.append(CroppingImg(
                            top_left=(x1, y1),
                            bottom_right=(x2, y2),
                            description=description,
                            save_path=""
                        ))
                        logs.info(f"Extracted region: {location}, coords: ({x1}, {y1}), ({x2}, {y2}), description: {description}")
        
        logs.info(f"Total suspicious regions extracted: {len(cropping_imgs)}")
        
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
