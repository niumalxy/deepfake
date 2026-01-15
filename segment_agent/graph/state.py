from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
import operator
from entity.segment_agent_status import AgentStatus
from PIL import Image

class AgentState(TypedDict):
    status: AgentStatus
    content_messages: Annotated[Sequence[BaseMessage], operator.add]
    analysis_messages: Annotated[Sequence[BaseMessage], operator.add]
    origin_img: Image.Image
    cropped_imgs: list[str] = []    # 存切割后的图像路径
    cropping_imgs: list[CroppingImg] = []
    current_img_idx: int = 0    # 当前正在处理的图像索引
    log_id: str = ""    # 日志id

class CroppingImg(TypedDict):
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]
    description: str
    save_path: str

class CroppedImg(TypedDict):
    save_path: str
    items: str
    description: str
    is_done: bool = False
    analysis_result: str = ""