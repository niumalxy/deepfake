from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
import operator
from entity.segment_agent_status import AgentStatus
from PIL import Image

class CroppingImg(TypedDict):
    items: str
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

class AgentState(TypedDict):
    status: Annotated[AgentStatus, lambda x, y: y]
    content_messages: Annotated[Sequence[BaseMessage], operator.add]
    analysis_messages: Sequence[BaseMessage]
    origin_img: Image.Image
    cropped_imgs: list[CroppedImg] = []    # 存切割后的图像路径
    cropping_imgs: list[CroppingImg] = []
    current_img_idx: Annotated[int, lambda x, y: y]  # 当前正在处理的图像索引
    report: str = ""
    prediction: str = ""
    tool_call_times: int = 0
    log_id: str = ""    # 日志id
