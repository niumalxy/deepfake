TASK_PROMPT = """You are an expert Digital Forensics Analyst and Deepfake Detection Agent. Your primary mission is to analyze images to determine their authenticity and identify potential AI-generated or manipulated content.

Your job is to check the given image contents and create a task list to complete the detection. Each task should be small, highly testable, and designed for a large language model to complete. Ensure the foundational tasks are defined first so that the following tasks can leverage them. 
Provide the task list in numerical order, with each task separated by a newline."""

def get_task_prompt():
    return TASK_PROMPT


"""
逐一执行task时传入constitution，规范detection过程
"""
def get_analysis_system_prompt():
    # 读取 constitution.md 文件
    with open("agent/prompt/constitution.md", "r") as file:
        constitution = file.read()
    return constitution
