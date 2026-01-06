ANALYSIS_USER_PROMPT = """
My Extra Requirements is following, you can pay attention to them but not strictly follow them:
{user_requirement}
"""

TASKS_USER_PROMPT = """My plan is following, you're supposed to follow it step by step:
{plan}
"""

ANALYSIS_EXECUTE_TASK_PROMPT = """
The task list is following:
{task_list}

You need to work with NO.{task_idx} task.
**IMPORTANT**: If you are certain that you have the current conclusion and complete this task, you should write "<complete>" at the end of your response.
"""
def get_analysis_user_prompt(user_requirement: str) -> str:
    if not user_requirement:
        return ""
    return ANALYSIS_USER_PROMPT.format(user_requirement=user_requirement)

def get_tasks_user_prompt(plan: str) -> str:
    if not plan:
        return ""
    return TASKS_USER_PROMPT.format(plan=plan)

def get_analysis_execute_task_prompt(task_list: list, task_idx: int) -> str:
    tasks = ""
    for task in task_list:
        tasks += f"- {task}\n"
    return ANALYSIS_EXECUTE_TASK_PROMPT.format(task_list=tasks, task_idx=task_idx)