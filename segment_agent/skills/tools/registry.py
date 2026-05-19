from agent.tools.__init__ import TOOLS_MAPPING as ORIGINAL_TOOLS_MAPPING
from typing import List, Any

def execute_image_skill_wrapper(skill_name: str, skill_params: dict, img=None):
    """
    Wrapper to execute a specific image skill by name.
    """
    if skill_name not in ORIGINAL_TOOLS_MAPPING:
        raise ValueError(f"Unknown skill: {skill_name}")

    tool_func = ORIGINAL_TOOLS_MAPPING[skill_name]

    if skill_params is None:
        skill_params = {}

    if img is not None:
        skill_params['img'] = img

    return tool_func(**skill_params)


def prune_image_context(messages: List[Any], max_images: int = 2) -> List[Any]:
    """
    确保对话历史中最多保留 max_images 张图像。
    超出部分用描述性元数据文本替代 image_url 块。

    Args:
        messages: LangChain 消息列表
        max_images: 最多保留的图像数（默认 2）

    Returns:
        裁剪后的消息列表
    """
    image_positions = []
    for mi, msg in enumerate(messages):
        content = getattr(msg, 'content', None)
        if isinstance(content, list):
            for bi, block in enumerate(content):
                if isinstance(block, dict) and block.get('type') == 'image_url':
                    image_positions.append((mi, bi))

    if len(image_positions) <= max_images:
        return messages

    positions_to_keep = set(image_positions[-max_images:])

    for mi, bi in image_positions:
        if (mi, bi) in positions_to_keep:
            continue

        content = messages[mi].content
        metadata_text = "[图像已省略以节省上下文]"

        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                txt = block.get('text', '')
                if '完整图像' in txt:
                    metadata_text = "[原始完整图像已省略，可调用 view_original_image 工具重新查看]"
                    break
                elif '待检测区域' in txt or '检测区域的图像' in txt:
                    metadata_text = "[局部区域图像已省略，区域描述信息已保留在上方文本中]"
                    break

        content[bi] = {
            "type": "text",
            "text": metadata_text
        }

    return messages


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_image_skill",
            "description": "Execute an image manipulation skill. Available skills: crop_img (top_left, bottom_right), resize_img (size), test_tool (dummy), rotate_img (angle, expand), flip_img (direction), adjust_brightness (factor), adjust_contrast (factor), adjust_saturation (factor), adjust_sharpness (factor), blur_img (radius), sharpen_img, convert_to_grayscale, convert_to_rgb, add_border (border_width, border_color), invert_img, pad_img (padding, color).",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the image skill to execute.",
                        "enum": [
                            "crop_img", "resize_img", "test_tool", "rotate_img", "flip_img",
                            "adjust_brightness", "adjust_contrast", "adjust_saturation", "adjust_sharpness",
                            "blur_img", "sharpen_img", "convert_to_grayscale", "convert_to_rgb",
                            "add_border", "invert_img", "pad_img"
                        ]
                    },
                    "skill_params": {
                        "type": "object",
                        "description": "The parameters for the specific skill as key-value pairs. E.g., {'top_left': [0, 0], 'bottom_right': [100, 100]} for crop_img. Provide {} if the skill requires no parameters."
                    }
                },
                "required": ["skill_name", "skill_params"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_original_image",
            "description": "查看完整的原始图像。当你需要对比当前局部区域与整体图像的关系、检查上下文一致性、或原始图像已被省略需要重新查看时，调用此工具获取原始完整图像。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

TOOLS_MAPPING = {
    "execute_image_skill": execute_image_skill_wrapper
}
