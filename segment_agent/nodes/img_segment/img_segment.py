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
