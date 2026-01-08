from agent.configuration.configuration import AgentConfiguration
from agent.graph.state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from logger import logs
from agent.img_content.prompt.analyze_content import *
from chat_model.openai.langchain_model import model
from entity.agent_status import AgentStatus
from db.local_map import status_map

def extract_img_content(state: AgentState, config: AgentConfiguration) -> str:
    logs.info("--- Extracting Image Content ---")
    state.update({"status": AgentStatus.ANALYZING})
    status_map[config.get('task_id', '')] = AgentStatus.ANALYZING

    content = [{
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{config.get('origin_img', '')}"
        }
    }]

    messages = [
        SystemMessage(content=get_analyze_content_prompt()),
        HumanMessage(content=content)
    ]
    # 模型输出
    response = model.invoke(messages)
    logs.info(f"img_content response: {response.content}")

    return {"messages": [response], "content": response.content}