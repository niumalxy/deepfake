from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from segment_agent.graph.state import AgentState
from entity.segment_agent_status import AgentStatus
from logger import logs
from chat_model.openai.langchain_model import model
from utils.img_convert import img_to_base64
from segment_agent.nodes.rag.prompt import get_rag_system_prompt
from segment_agent.skills.tools.rag_tool import get_rag_tool

def rag_node(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAG节点：使用FAISS检索相似历史数据，供Agent做决策上下文
    
    Args:
        state: AgentState包含当前工作流状态
        config: 配置对象
        
    Returns:
        Dict[str, Any]: 更新后的状态
    """
    logs.info("--- RAG Node: Retrieving Historical Case Context ---")
    
    origin_img = state.get('origin_img')
    if not origin_img:
        logs.error("No origin image found in State for RAG node to embed.")
        return {
            "status": AgentStatus.INVALID
        }
    
    # 获取动态tool
    rag_tool = get_rag_tool(origin_img)
    tools = [rag_tool]
    agent = model.bind_tools(tools)
    
    # 第一次进入这个节点
    if not state.get("rag_messages"):
        logs.info("Initializing RAG interactions...")
        
        img_base64 = img_to_base64(origin_img)
        user_content = [
            {
                "type": "text",
                "text": "This is the image we are analyzing. Please query the FAISS database to find similar historical cases."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            }
        ]
        
        system_prompt = get_rag_system_prompt(config.get('use_chinese', False))
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
    else:
        # 已有交互历史（可能刚调用完tool返回）
        messages = list(state["rag_messages"])
        
    try:
        response = agent.invoke(messages)
        
        # 提取当前上下文 summary 用于后续节点
        retrieved_context = state.get("retrieved_context", "")
        # 如果模型直接回答了一段文本，把它拼进 context
        if response.content and isinstance(response.content, str):
            retrieved_context += f"\n{response.content}"
        
        return {
            "rag_messages": [response],
            "retrieved_context": retrieved_context,
            "status": AgentStatus.WAITING
        }
        
    except Exception as e:
        logs.error(f"Error in RAG node: {e}")
        return {
            "retrieved_context": "Failed to retrieve historical context due to an error.",
            "status": AgentStatus.INVALID
        }
