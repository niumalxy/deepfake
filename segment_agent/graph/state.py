from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from entity.agent_status import AgentStatus

class AgentState(TypedDict):
    status: AgentStatus
    messages: Annotated[Sequence[BaseMessage], operator.add]