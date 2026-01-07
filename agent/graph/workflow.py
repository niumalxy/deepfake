from langgraph.graph import StateGraph, END, START
from .state import AgentState
from agent.img_content.work import extract_img_content
from agent.analysis.work import analyze_content, generate_tasks
from agent.plan.work import plan
from agent.configuration.configuration import AgentConfiguration
import functools

from agent.summary.work import summarize

def analysis_next_node(state: AgentState):
    messages = state['analysis_messages']
    last_message = messages[-1]
    if "<complete>" in last_message.content.lower() and state.get("current_task") == len(state.get("tasks")):
        return "complete"
    return "continue"


def create_graph(task_id: str, img: str = "", use_chinese: bool = False):
    config = AgentConfiguration(task_id=task_id, origin_img=img, use_chinese=use_chinese)
    workflow = StateGraph(AgentState)
    
    # Add nodes
    # Use functools.partial to inject the specific config object into nodes that require it
    workflow.add_node("img_content", functools.partial(extract_img_content, config=config))
    workflow.add_node("plan", functools.partial(plan, config=config))
    workflow.add_node("tasks", functools.partial(generate_tasks, config=config))
    workflow.add_node("analysis", functools.partial(analyze_content, config=config))
    workflow.add_node("summary", functools.partial(summarize, config=config))

    # Set entry point
    workflow.set_entry_point("img_content")

    # Add edges
    workflow.add_edge("img_content", "plan")
    workflow.add_edge("plan", "tasks")
    workflow.add_edge("tasks", "analysis")
    workflow.add_conditional_edges("analysis", analysis_next_node, {
        "continue": "analysis",
        "complete": "summary"
    })
    workflow.add_edge("summary", END)

    return workflow.compile()
