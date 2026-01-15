from segment_agent.skills.tools.sr_tool.sr_tool import super_resolution
from segment_agent.skills.tools.img_tool import (
    crop_img,
    resize_img,
    rotate_img,
    flip_img,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_sharpness,
    blur_img,
    sharpen_img,
    convert_to_grayscale,
    convert_to_rgb,
    add_border,
    invert_img,
    pad_img
)

# Tool definitions
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "super_resolution",
            "description": "Enhance image resolution using super-resolution models. Supports both local and remote modes. Local mode uses OpenCV DNN models (EDSR, ESPCN, FSRCNN, LAPSRN) with bicubic interpolation as fallback. Remote mode calls external API services. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scale": {
                        "type": "integer",
                        "description": "Upscaling factor for the image resolution enhancement. Default is 4.",
                        "default": 4,
                        "minimum": 1,
                        "maximum": 8
                    },
                },
                "required": []
            }
        }
    },
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
            "name": "rotate_img",
            "description": "Rotate the image by the specified angle in degrees (counter-clockwise). The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees (counter-clockwise). Default is 90.",
                        "default": 90
                    },
                    "expand": {
                        "type": "boolean",
                        "description": "If True, expands the output image to fit the rotated image. Default is False.",
                        "default": False
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flip_img",
            "description": "Flip the image horizontally or vertically. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Flip direction - 'horizontal' or 'vertical'. Default is 'horizontal'.",
                        "enum": ["horizontal", "vertical"],
                        "default": "horizontal"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_brightness",
            "description": "Adjust the brightness of the image. A factor of 1.0 gives the original image, 0.0 gives a black image, and 2.0 gives twice the brightness. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": "Brightness factor. 1.0 gives original image, 0.0 gives black image, 2.0 gives twice brightness. Default is 1.0.",
                        "default": 1.0,
                        "minimum": 0
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_contrast",
            "description": "Adjust the contrast of the image. A factor of 1.0 gives the original image, 0.0 gives a solid gray image, and 2.0 gives twice the contrast. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": "Contrast factor. 1.0 gives original image, 0.0 gives solid gray image, 2.0 gives twice contrast. Default is 1.0.",
                        "default": 1.0,
                        "minimum": 0
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_saturation",
            "description": "Adjust the saturation of the image. A factor of 1.0 gives the original image, 0.0 gives a grayscale image, and 2.0 gives twice the saturation. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": "Saturation factor. 1.0 gives original image, 0.0 gives grayscale, 2.0 gives twice saturation. Default is 1.0.",
                        "default": 1.0,
                        "minimum": 0
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_sharpness",
            "description": "Adjust the sharpness of the image. A factor of 1.0 gives the original image, 0.0 gives a blurred image, and 2.0 gives a sharpened image. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {
                        "type": "number",
                        "description": "Sharpness factor. 1.0 gives original image, 0.0 gives blurred image, 2.0 gives sharpened image. Default is 1.0.",
                        "default": 1.0,
                        "minimum": 0
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "blur_img",
            "description": "Apply Gaussian blur to the image. A larger radius gives more blur effect. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "Blur radius. Larger radius gives more blur. Default is 2.",
                        "default": 2,
                        "minimum": 0
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sharpen_img",
            "description": "Apply a sharpening filter to the image to enhance edges and details. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_to_grayscale",
            "description": "Convert the image to grayscale (black and white). The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_to_rgb",
            "description": "Convert the image to RGB mode. Useful for converting grayscale or other color modes to RGB. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_border",
            "description": "Add a border around the image with specified width and color. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "border_width": {
                        "type": "number",
                        "description": "Width of the border in pixels. Default is 10.",
                        "default": 10,
                        "minimum": 0
                    },
                    "border_color": {
                        "type": "string",
                        "description": "Color of the border. Can be a color name (e.g., 'black', 'red'), hex code (e.g., '#FF0000'), or RGB tuple. Default is 'black'.",
                        "default": "black"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "invert_img",
            "description": "Invert the colors of the image (create a negative effect). The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pad_img",
            "description": "Add padding around the image with specified size and color. The image context is handled automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "padding": {
                        "type": "number",
                        "description": "Padding size in pixels. Can be a single value or tuple (left, top, right, bottom). Default is 10.",
                        "default": 10,
                        "minimum": 0
                    },
                    "color": {
                        "type": "string",
                        "description": "Padding color. Can be a color name (e.g., 'white', 'gray'), hex code (e.g., '#FFFFFF'), or RGB tuple. Default is 'white'.",
                        "default": "white"
                    }
                },
                "required": []
            }
        }
    }
]

TOOLS_MAPPING = {
    "super_resolution": super_resolution,
    "crop_img": crop_img,
    "resize_img": resize_img,
    "rotate_img": rotate_img,
    "flip_img": flip_img,
    "adjust_brightness": adjust_brightness,
    "adjust_contrast": adjust_contrast,
    "adjust_saturation": adjust_saturation,
    "adjust_sharpness": adjust_sharpness,
    "blur_img": blur_img,
    "sharpen_img": sharpen_img,
    "convert_to_grayscale": convert_to_grayscale,
    "convert_to_rgb": convert_to_rgb,
    "add_border": add_border,
    "invert_img": invert_img,
    "pad_img": pad_img
}
