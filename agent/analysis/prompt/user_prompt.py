USER_PROMPT = """
My Extra Requirements is following, you can pay attention to them but not strictly follow them:
{user_requirement}
"""

def get_user_prompt(user_requirement: str) -> str:
    if not user_requirement:
        return ""
    return USER_PROMPT.format(user_requirement=user_requirement)