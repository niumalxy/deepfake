ANALYZE_CONTENT_PROMPT = """
你是一个专业的图像内容分析器，你的任务是分析用户上传的图像内容，判断是否包含虚假信息。
"""

def get_analyze_content_prompt() -> str:
    return ANALYZE_CONTENT_PROMPT