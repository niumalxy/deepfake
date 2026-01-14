from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
import operator
from entity.agent_status import AgentStatus

class AgentState(TypedDict):
    status: AgentStatus
    messages: Annotated[Sequence[BaseMessage], operator.add]
    origin_img: Image.Image
    cropped_imgs: list[Image.Image] = []
    cropping_imgs: list[Any] = []
