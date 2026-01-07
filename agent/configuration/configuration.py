from langchain_core.runnables import RunnableConfig

class AgentConfiguration(RunnableConfig):
    """The configuration for the agent."""
    task_id: str
    origin_img: str
    use_chinese: bool