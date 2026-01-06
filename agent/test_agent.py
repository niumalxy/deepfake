
import sys
import os
sys.path.append("./")

from agent.graph import create_graph
from langchain_core.messages import HumanMessage
from agent.graph.state import AgentState
import utils.img_convert as img_utils

def run_test():
    task_id = "test_run"
    img = img_utils.img_path_to_base64('./test.png')

    print(f"Creating graph for task_id: {task_id}")
    graph = create_graph(task_id=task_id, img=img)
    
    # Mock input
    # Assuming we have a dummy image or can pass a path. 
    # For testing without a real image model, we might need to mock the model response or ensure the LLM key is set.
    # Since I cannot easily verify LLM key, I'll assume the environment is set up.
    
    inputs = {
        "user_input": "Test request to analyze image.",
        "image": "d:/yan/毕设/deepfake/test_image.jpg", # Placeholder path
        "messages": [HumanMessage(content="Start analysis")]
    }

    print("Running Agent Graph...")
    try:
            
        result = graph.invoke(inputs)
        print("\n--- Result ---")
        print(f"Plan: {result.get('plan')}")
        print(f"Tasks: {result.get('tasks')}")
        for message in result.get('analysis_messages', []):
            print(f"[{message.type}]: {message.content}")
            
    except Exception as e:
        print(f"Error running graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
