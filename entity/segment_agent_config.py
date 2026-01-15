from pydantic import BaseModel

class SegmentAgentConfig(BaseModel):
    task_id: str
    
