from enum import Enum

class AgentStatus(Enum):
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    REPORTING = "reporting"
    FINISHED = "finished"