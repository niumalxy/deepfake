from PIL import Image
from segment_agent.graph.state import AgentState, CroppedImg
from entity.segment_agent_status import AgentStatus
import os
from logger import logs



def crop_image_by_coords(state: AgentState, config: dict):
    """
    根据坐标裁剪图像并保存
    
    Args:
        state: AgentState，包含origin_img和cropping_imgs
        config: 配置字典，包含task_id等配置信息
    
    Returns:
        AgentState: 更新后的state
    """
    
    origin_img = state['origin_img']
    origin_save_path = "./segment_agent/nodes/img_content/docs"
    if not os.path.exists(origin_save_path):
        os.makedirs(origin_save_path, exist_ok=True)
    task_id = config.get('task_id', 'default')
    
    # 确保保存目录存在
    os.makedirs(origin_save_path, exist_ok=True)
    
    cropped_imgs = []
    cropping_imgs = state.get('cropping_imgs', [])
    logs.info(f"开始裁剪图像，共{len(cropping_imgs)}张")

    for idx, item in enumerate(cropping_imgs):
        x1, y1 = item.get('top_left', (0, 0))
        x2, y2 = item.get('bottom_right', (0, 0))
        items = item.get('items', '')
        description = item.get('description', '')

        # 获取图像尺寸
        width, height = origin_img.size
        
        # 坐标规范化：确保 x1 < x2, y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 边界检查与修正
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # 检查有效性：如果裁剪区域为空（宽度或高度为0），则跳过或使用默认处理
        if x2 <= x1 or y2 <= y1:
            logs.warning(f"Invalid crop area for item {idx}: ({x1}, {y1}, {x2}, {y2}). Skipping.")
            continue
            
        box = (x1, y1, x2, y2)
        img = origin_img.crop(box)
        
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        save_path = os.path.join(origin_save_path, f"{task_id}_{idx}.jpg")
        img.save(save_path)
        
        # 创建CroppedImg对象
        cropped_img = CroppedImg(
            save_path=save_path,
            items=items,
            description=description,
            is_done=False,
            analysis_result=""
        )
        cropped_imgs.append(cropped_img)
    
    return {
        "cropped_imgs": cropped_imgs,
        "status": AgentStatus.ANALYZING,
    }
