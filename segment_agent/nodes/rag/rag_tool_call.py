import json
from langchain_core.messages import ToolMessage
from segment_agent.graph.state import AgentState
from logger import logs
from segment_agent.skills.tools.rag_tool import get_rag_tool
from typing import Dict, Any

def rag_tool_call(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """执行 RAG Node 请求的工具调用"""
    logs.info("--- RAG Tool Call Execution ---")
    
    last_message = state["rag_messages"][-1]
    origin_img = state.get('origin_img')
    
    rag_tool = get_rag_tool(origin_img)
    tool_map = {
        "search_similar_images": rag_tool
    }

    tool_responses = []
    
    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        logs.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        if tool_name in tool_map:
            try:
                # 显式执行 tool
                result = tool_map[tool_name].invoke(tool_args)
                logs.info(f"Tool {tool_name} executed successfully. Length: {len(str(result))}")
            except Exception as e:
                result = f"Error executing tool {tool_name}: {str(e)}"
                logs.error(result)
        else:
            result = f"Error: Tool {tool_name} not found"
            logs.error(result)
            
        tool_responses.append(ToolMessage(
            tool_call_id=tc["id"],
            name=tool_name,
            content=str(result)
        ))

    return {
        "rag_messages": tool_responses,
        "tool_call_times": state.get("tool_call_times", 0) + 1
    }
