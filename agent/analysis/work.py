from agent.analysis.prompt.user_prompt import get_analysis_user_prompt
from agent.analysis.prompt.user_prompt import get_tasks_user_prompt
from agent.graph.state import AgentState
from chat_model.openai.langchain_model import model
from langchain_core.messages import HumanMessage, SystemMessage
from agent.configuration.configuration import AgentConfiguration
from logger import logs
from agent.analysis.prompt.system_prompt import *
from agent.analysis.prompt.user_prompt import *
from entity.agent_status import AgentStatus

def generate_tasks(state: AgentState, config: AgentConfiguration):
    """
    Generate tasks using the LLM.
    """
    logs.info("--- Generating Tasks ---")
    # 读取plan
    with open(f"agent/plan/docs/plan_{config.get('task_id', '')}.md", "r", encoding="utf-8") as f:
        plan = f.read()
    messages = [SystemMessage(content=get_task_prompt()), HumanMessage(content=get_tasks_user_prompt(plan))]
    response = model.invoke(messages)
    # 清除空字符串
    origin_plan = response.content.split('\n')
    plan = []
    for item in origin_plan:
        if item.strip():
            plan.append(item.strip())
    logs.info(f"Plan: {plan}")
    return {
        "status": AgentStatus.ANALYZING,
        "tasks": plan,
        "current_task": 0,
    }

def analyze_content(state: AgentState, config: AgentConfiguration):
    """
    Analyze the extracted content using the LLM.
    """
    logs.info("--- Analyzing Content ---")
    # state.update({"status": AgentStatus.EXECUTING}) -> Will return at end
    user_input = state.get("user_input", "")
    logs.info(f"Total step:{len(state.get('tasks', []))}, current step: {state.get('current_task', 0)+1}")
    content = []
    content.append({"type": "text", "text": get_analysis_user_prompt(user_input)})
    if not state.get("analysis_messages", []):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{config.get('origin_img', '')}"
            }
        })
        new_messages = [SystemMessage(content=get_analysis_system_prompt()), HumanMessage(content=content)]
        messages = new_messages
    else:
        new_messages = [HumanMessage(content=content)]
        messages = state.get("analysis_messages") + new_messages
    messages.append(HumanMessage(content=get_analysis_execute_task_prompt(state.get("tasks", []), state.get("current_task", 0))))
    
    response = model.invoke(messages)
    
    logs.info(f"analysis response: {response.content}")

    updates = {
        "status": AgentStatus.EXECUTING,
        "messages": [response],
        "analysis_messages": new_messages + [response]
    }

    if "<complete>" in response.content.lower():
        current_task = state.get("current_task", 0)
        updates["current_task"] = current_task + 1
        
    return updates