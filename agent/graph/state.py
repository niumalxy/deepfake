from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from entity.agent_status import AgentStatus

class AgentState(TypedDict):
    status: AgentStatus
    messages: Annotated[Sequence[BaseMessage], operator.add]
    analysis_messages: Annotated[Sequence[BaseMessage], operator.add]
    report_messages: Annotated[Sequence[BaseMessage], operator.add]
    tasks: list[str]
    content: str
    current_task: int = 0
    plan: str
    user_input: str
    image: str
    origin_image: str
