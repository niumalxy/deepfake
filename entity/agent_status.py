from enum import Enum

class AgentStatus(Enum):
    WAITING = "waiting"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    REPORTING = "reporting"
    FINISHED = "finished"