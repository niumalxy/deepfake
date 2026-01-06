from agent.configuration.configuration import AgentConfiguration
from agent.graph.state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from logger import logs
from chat_model.openai.langchain_model import model
import agent.plan.prompt.plan_prompt as prompts
from entity.agent_status import AgentStatus

def plan(state: AgentState, config: AgentConfiguration):
    logs.info("--- Planning Tasks ---")
    state.update({"status": AgentStatus.PLANNING})
    # Get the image content analysis from the state
    content_analysis = state.get("content", "")
    content = []
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{config.get('origin_img', '')}"
        }
    })
    content.append({"type": "text", "text": f"This is the image content analyst's analysis of the image content, structured data as follows:\n{content_analysis}"})
    messages = [
        SystemMessage(content=prompts.get_plan_prompt()),
        HumanMessage(content=content)
    ]
    
    response = model.invoke(messages)
    # task保存至本地
    with open(f"agent/plan/docs/plan_{config.get('task_id', 'default')}.md", "w", encoding="utf-8") as f:
        f.write(response.content)
