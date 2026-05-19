

from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from segment_agent.graph.state import AgentState
from segment_agent.skills.tools.registry import TOOLS_SCHEMA, TOOLS_MAPPING, prune_image_context
from chat_model.openai.langchain_model import model
from entity.segment_agent_status import AgentStatus
from logger import logs
from PIL import Image
import utils.img_convert as utils



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
    
    analysis_messages = state.get('analysis_messages', [])
    if not analysis_messages:
        logs.warning("No analysis messages to process for tool call")
        return {"status": AgentStatus.ANALYZING}
    
    last_ai_message = None
    for msg in reversed(analysis_messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message or not hasattr(last_ai_message, 'tool_calls') or not last_ai_message.tool_calls:
        logs.warning("No tool calls found in the messages")
        return {"analysis_messages": analysis_messages}

    tool_call_times = state.get("tool_call_times", 0) + 1
    if tool_call_times > 1:
        logs.warning("Tool call times exceeds 1 times, stop calling tools")
        return {"analysis_messages": analysis_messages + [SystemMessage(content="警告：工具调用不得超过1次，请勿重复调用tool_calls。")]}

    updated_messages = list(analysis_messages)
    current_idx = state.get('current_img_idx', 0)

    for tool_call in last_ai_message.tool_calls:
        function_name = tool_call['name']
        function_args = tool_call['args'].copy()
        logs.info(f"Calling tool: {function_name} with args: {function_args}")

        if function_name == "view_original_image":
            try:
                origin_img = state.get('origin_img')
                if origin_img is None:
                    tool_result = "错误：无法获取原始图像"
                else:
                    origin_base64 = utils.img_to_base64(origin_img)
                    tool_result = [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{origin_base64}"
                        }
                    }, {
                        "type": "text",
                        "text": f"这是完整的原始图像（尺寸：{origin_img.size[0]}x{origin_img.size[1]}），供你对比当前局部区域。"
                    }]

                tool_message = ToolMessage(
                    content=tool_result,
                    name=function_name,
                    tool_call_id=tool_call['id']
                )
                updated_messages.append(tool_message)

                updated_messages = prune_image_context(updated_messages, max_images=2)
                logs.info(f"Successfully executed tool: {function_name}")
            except Exception as e:
                logs.error(f"Error executing tool {function_name}: {e}")
                error_message = ToolMessage(
                    content=f"Error executing {function_name}: {str(e)}",
                    name=function_name,
                    tool_call_id=tool_call['id']
                )
                updated_messages.append(error_message)
        elif function_name in TOOLS_MAPPING:
            try:
                function_args["img"] = Image.open(state["cropped_imgs"][current_idx]["save_path"])
                tool_function = TOOLS_MAPPING[function_name]
                tool_result = tool_function(**function_args)
                
                if type(tool_result) == Image.Image:
                    display_args = {k: v for k, v in function_args.items() if k != "img"}
                    tool_result = [{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{utils.img_to_base64(tool_result)}"
                        }
                    }, {
                        "type": "text",
                        "text": f"[{function_name} 处理后的图像，参数：{display_args}]"
                    }]
                else:
                    tool_result = str(tool_result)

                tool_message = ToolMessage(
                    content=tool_result,
                    name=function_name,
                    tool_call_id=tool_call['id']
                )
                updated_messages.append(tool_message)

                if isinstance(tool_result, list):
                    updated_messages = prune_image_context(updated_messages, max_images=2)
                
                logs.info(f"Successfully executed tool: {function_name}")
            except Exception as e:
                logs.error(f"Error executing tool {function_name}: {e}")
                error_message = ToolMessage(
                    content=f"Error executing {function_name}: {str(e)}",
                    name=function_name,
                    tool_call_id=tool_call['id']
                )
                updated_messages.append(error_message)
        else:
            logs.warning(f"Tool {function_name} not found in TOOLS_MAPPING")
            error_message = ToolMessage(
                content=f"Tool {function_name} not available",
                name=function_name,
                tool_call_id=tool_call['id']
            )
            updated_messages.append(error_message)
        
    bound_model = model.bind_tools(TOOLS_SCHEMA)
    response = bound_model.invoke(updated_messages)
    updated_messages.append(response)
    
    return {
        "tool_call_times": tool_call_times,
        "analysis_messages": updated_messages,
        "status": AgentStatus.ANALYZING
    }
