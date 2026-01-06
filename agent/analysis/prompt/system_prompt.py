"""
初始，先根据图像内容自行安排分析tasks
"""
TASKS_PROMPT = """
You are an expert Digital Forensics Analyst and Deepfake Detection Agent. Your primary mission is to analyze images to determine their authenticity and identify potential AI-generated or manipulated content.

This is the image content analyst's analysis of the image content, structured data as follows:
{img_contents}

Your job is to check the given image contents and create a task list to complete the detection. Each task should be small, highly testable, and designed for a large language model to complete. Ensure the foundational tasks are defined first so that the following tasks can leverage them. Provide the task list in markdown format.
"""

"""
逐一执行task时传入constitution，规范detection过程
"""
# 读取 constitution.md 文件
with open("agent/prompt/constitution.md", "r") as file:
    constitution = file.read()



