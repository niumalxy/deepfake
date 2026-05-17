from typing import TypedDict, Any, Optional
from PIL import Image

class SegmentAgentConfig(TypedDict, total=False):
    task_id: str
    img: Image.Image
    use_chinese: bool
    label: Any
    need_rag: bool
    skip_dump: bool
        
