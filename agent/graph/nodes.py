from langchain_core.messages import HumanMessage
from .state import AgentState
from chat_model.openai.langchain_model import model

def call_model(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}
