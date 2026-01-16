

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from segment_agent.graph.state import AgentState
from segment_agent.skills.tools.registry import TOOLS_SCHEMA, TOOLS_MAPPING
from chat_model.openai.langchain_model import model
from entity.segment_agent_status import AgentStatus
from logger import logs
from PIL import Image

def tool_call(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    工具调用节点 - 执行AI请求的工具调用
    
    Args:
        state: AgentState，包含当前工作流状态
        config: 配置字典，包含模型等信息
    
    Returns:
        Dict[str, Any]: 更新后的状态字段
    """
    logs.info("--- Executing Tool Call ---")
    
    # 获取最新的分析消息
    analysis_messages = state.get('analysis_messages', [])
    if not analysis_messages:
        logs.warning("No analysis messages to process for tool call")
        return {"status": AgentStatus.ANALYZING}
    
    # 获取最后一条AI消息
    last_ai_message = None
    for msg in reversed(analysis_messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message or not hasattr(last_ai_message, 'tool_calls') or not last_ai_message.tool_calls:
        logs.warning("No tool calls found in the messages")
        # 返回原状态继续分析
        return {"analysis_messages": analysis_messages}
    
    # 执行所有找到的工具调用
    updated_messages = analysis_messages.copy()
    for tool_call in last_ai_message.tool_calls:
        function_name = tool_call['name']
        function_args = tool_call['args']
        current_idx = state.get('current_img_idx', 0)
        logs.info(f"Calling tool: {function_name} with args: {function_args}")
        function_args["img"] = Image.open(state["cropped_imgs"][current_idx]["save_path"])
        if function_name in TOOLS_MAPPING:
            try:
                # 执行工具函数
                tool_function = TOOLS_MAPPING[function_name]
                tool_result = tool_function(**function_args)
                
                # 创建工具消息
                tool_message = ToolMessage(
                    content=str(tool_result),
                    name=function_name,
                    tool_call_id=tool_call['id']
                )
                
                # 添加工具消息到消息列表
                updated_messages.append(tool_message)
                
                logs.info(f"Successfully executed tool: {function_name}")
            except Exception as e:
                logs.error(f"Error executing tool {function_name}: {e}")
                
                # 创建错误消息
                error_message = ToolMessage(
                    content=f"Error executing {function_name}: {str(e)}",
                    name=function_name,
                    tool_call_id=tool_call['id']
                )
                
                updated_messages.append(error_message)
        else:
            logs.warning(f"Tool {function_name} not found in TOOLS_MAPPING")
            
            # 创建未找到工具的消息
            error_message = ToolMessage(
                content=f"Tool {function_name} not available",
                name=function_name,
                tool_call_id=tool_call['id']
            )
            
            updated_messages.append(error_message)
    
    # 调用模型获取新的AI响应
    bound_model = model.bind_tools(TOOLS_SCHEMA)
    response = bound_model.invoke(updated_messages)
    
    # 添加AI的新响应
    updated_messages.append(response)
    
    return {
        "analysis_messages": updated_messages,
        "status": AgentStatus.ANALYZING  # 继续分析流程
    }