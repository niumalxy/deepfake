from segment_agent.skills.tools.sr_tool.sr_tool import super_resolution

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
]

TOOLS_MAPPING = {
    "super_resolution": super_resolution,
}
