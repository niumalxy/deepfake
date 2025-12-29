USER_PROMPT = """
我的要求如下：{user_requirement}
"""

def get_user_prompt(user_requirement: str) -> str:
    if not user_requirement:
        return ""
    return USER_PROMPT.format(user_requirement=user_requirement)