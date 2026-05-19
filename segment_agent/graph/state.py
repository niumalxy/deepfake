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
    anomaly_type: str  # splicing_edge | repeated_pattern | anatomical | physics | text_gibberish | other

class CroppedImg(TypedDict):
    save_path: str
    items: str
    description: str
    is_done: bool = False
    analysis_result: str = ""
    # Phase 3 structured fields populated by next_part from <region_verdict>
    verdict: str = ""           # Real | Likely Real | Uncertain | Likely Fake | Fake
    confidence: int = 0          # 0-100
    strong_count: int = 0
    weak_count: int = 0
    anomaly_type: str = "other"  # carried over from screening
    top_left: tuple[int, int] = (0, 0)
    bottom_right: tuple[int, int] = (0, 0)

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
    rag_messages: Annotated[Sequence[BaseMessage], operator.add]
    retrieved_context: str = ""
    tool_call_times: int = 0
    analysis_iter_count: int = 0
    log_id: str = ""    # 日志id
