from agent.graph.state import AgentState
from agent.configuration.configuration import AgentConfiguration
from chat_model.openai.langchain_model import model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from logger import logs
from agent.summary.prompt.system_prompt import get_summary_system_prompt
from entity.agent_status import AgentStatus
import os
from db.local_map import status_map

def summarize(state: AgentState, config: AgentConfiguration):
    """
    Summarize the analysis results into a markdown report.
    """
    logs.info("--- Summarizing Results ---")
    status_map[config.get('task_id', '')] = AgentStatus.REPORTING

    # Extract analysis messages
    analysis_messages = state.get("analysis_messages", [])
    if not analysis_messages:
        logs.warning("No analysis messages found to summarize.")
        return {"status": AgentStatus.COMPLETED}

    # Convert messages to a string representation for the LLM
    # We focus on the content of the messages
    messages = [
        SystemMessage(content=get_summary_system_prompt(config.get('use_chinese', False))),
        HumanMessage(content=f"Here is the analysis log:\n\n")
    ]
    for msg in analysis_messages:
        # 只看模型回复
        if not isinstance(msg, AIMessage):
            continue
        messages.append(HumanMessage(content=f" {msg.content}\n\n"))



    response = model.invoke(messages)
    report_content = response.content

    logs.info("Summary generated successfully.")

    # Save the report to a file
    task_id = config.get('task_id', 'default')
    output_dir = "agent/summary/docs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/summary_{task_id}.md"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logs.info(f"Report saved to {file_path}")
    status_map[config.get('task_id', '')] = AgentStatus.COMPLETED
    return {
        "status": AgentStatus.COMPLETED,
        "report_messages": [response] # Optional: store report in state
    }
