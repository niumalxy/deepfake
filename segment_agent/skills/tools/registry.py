from agent.tools.__init__ import TOOLS_MAPPING as ORIGINAL_TOOLS_MAPPING

def execute_image_skill_wrapper(skill_name: str, skill_params: dict, img=None):
    """
    Wrapper to execute a specific image skill by name.
    """
    if skill_name not in ORIGINAL_TOOLS_MAPPING:
        raise ValueError(f"Unknown skill: {skill_name}")
    
    # Execute the requested skill
    tool_func = ORIGINAL_TOOLS_MAPPING[skill_name]
    
    # Ensure skill_params is a dictionary
    if skill_params is None:
        skill_params = {}
    
    # If the tool requires the img parameter, pass it uniformly
    if img is not None:
        skill_params['img'] = img
        
    return tool_func(**skill_params)

# Compacted TOOLS_SCHEMA to save token usage, utilizing "Claude Skills" approach
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
    }
]

TOOLS_MAPPING = {
    "execute_image_skill": execute_image_skill_wrapper
}
