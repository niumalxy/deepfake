from enum import Enum
from pydantic import BaseModel

class AgentStatus(Enum):
    INITIATING = "init"
    PLANNING = "planning"
    ANALYZING = "analyzing"
    TOOL_EXECUTING = "executing_tool"
    REFLECTING = "reflecting"
    REPORTING = "reporting"
    FINISHED = "finished"
    COMPLETED = "completed"

