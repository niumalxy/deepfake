from PIL import Image
from segment_agent.graph.workflow import create_graph
from logger import logs, scoped_context


def test_segment_agent():
    """
    测试segment_agent的完整流程
    """
    # 设置日志上下文
    ctx = {"log_id": "test_001"}
    
    with scoped_context(ctx):
        logs.info("Starting segment agent test")
        
        # 加载测试图像
        try:
            test_img = Image.open("test.png")
            logs.info(f"Loaded test image: {test_img.size}")
        except Exception as e:
            logs.error(f"Failed to load test image: {e}")
            return
        
        # 创建工作流图
        task_id = "test_task_001"
        graph = create_graph(task_id, test_img)
        
        # 初始化状态
        initial_state = {
            "status": None,
            "content_messages": [],
            "analysis_messages": [],
            "origin_img": test_img,
            "cropped_imgs": [],
            "cropping_imgs": [],
            "log_id": task_id,
            "current_analysis_idx": 0
        }
        
        # 执行工作流
        try:
            logs.info("Starting workflow execution")
            result = graph.invoke(initial_state)
            
            logs.info("Workflow execution completed")
            logs.info(f"Final status: {result.get('status')}")
            logs.info(f"Number of suspicious regions: {len(result.get('cropping_imgs', []))}")
            logs.info(f"Number of cropped images: {len(result.get('cropped_imgs', []))}")
            
            # 输出分析结果
            for idx, cropped_img in enumerate(result.get('cropped_imgs', [])):
                logs.info(f"\n--- Analysis Result {idx + 1} ---")
                logs.info(f"Save path: {cropped_img.get('save_path')}")
                logs.info(f"Description: {cropped_img.get('description')}")
                logs.info(f"Is done: {cropped_img.get('is_done')}")
                logs.info(f"Analysis result: {cropped_img.get('analysis_result', 'N/A')}")
            
        except Exception as e:
            logs.error(f"Error during workflow execution: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_segment_agent()
