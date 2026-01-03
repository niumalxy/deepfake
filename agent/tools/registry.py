from agent.tools.img_tool import crop_img, resize_img
from agent.tools.test_tool import test_tool

# Tool definitions
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "crop_img",
            "description": "Crop the image based on top-left and bottom-right coordinates. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_left": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Top-left coordinates (x, y)",
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "bottom_right": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Bottom-right coordinates (x, y)",
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["top_left", "bottom_right"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resize_img",
            "description": "Resize the image to the specified size. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "size": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Target size (width, height)",
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool to verify tool calling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dummy": {
                        "type": "string",
                        "description": "A dummy parameter to satisfy schema requirements."
                    }
                },
                "required": ["dummy"]
            }
        }
    }
]

TOOLS_MAPPING = {
    "crop_img": crop_img,
    "resize_img": resize_img,
    "test_tool": test_tool
}
