from typing import TypedDict, Any
from PIL import Image

class SegmentAgentConfig(TypedDict):
    task_id: str
    img: Image.Image
    use_chinese: bool
    label: Any
        
