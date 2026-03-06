from chat_model.openai.langchain_model import model
from langchain_core.messages import HumanMessage, SystemMessage
from db.mongodb import get_analysis_by_task_id
from entity.dump_type import DumpType
from logger import logs
import os
import shutil
import time

def backup_file(filepath: str) -> str:
    """Create a backup of the given file."""
    if not os.path.exists(filepath):
        return None
    
    dir_name = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    name, ext = os.path.splitext(base_name)
    timestamp = int(time.time())
    
    backup_path = os.path.join(dir_name, f"{name}_backup_{timestamp}{ext}")
    shutil.copy2(filepath, backup_path)
    logs.info(f"Backed up {filepath} to {backup_path}")
    return backup_path

def reflect_prompt(task_id: str, dump_type: str):
    """
    Perform reflection on an incorrect task prediction to update the prompt.
    """
    logs.info(f"--- Starting Reflection for task {task_id} (Type: {dump_type}) ---")
    
    analysis = get_analysis_by_task_id(task_id)
    if not analysis:
        logs.error(f"Reflection skipped: Analysis not found for task_id {task_id}")
        return
        
    origin_label = analysis.get("label")
    origin_description = analysis.get("description", "No description provided.")
    wrong_prediction = analysis.get("prediction")
    generated_report = analysis.get("report", "No report generated.")
    
    true_label_str = "fake" if origin_label == 0 else "normal"
    
    if dump_type == DumpType.SEGMENT.value or dump_type == DumpType.SEGMENT:
        prompt_file = "segment_agent/docs/segment_constitution.md"
        agent_type = "Segment Analysis Agent"
    elif dump_type == DumpType.REPORT.value or dump_type == DumpType.REPORT:
        prompt_file = "segment_agent/docs/constitution.md"
        agent_type = "Final Report Agent"
    else:
        logs.error(f"Unknown dump type for reflection: {dump_type}")
        return

    if not os.path.exists(prompt_file):
        logs.error(f"Prompt file not found: {prompt_file}")
        return
        
    with open(prompt_file, "r", encoding="utf-8") as f:
        current_prompt = f.read()
        
    reflection_system_prompt = f"""You are an Expert Prompt Engineer and Agent Behavior Analyst.
Your goal is to improve the instructions of a {agent_type} that recently made a classification error.

Below is the CURRENT PROMPT of the agent:
<current_prompt>
{current_prompt}
</current_prompt>

The agent was given an image with the TRUE LABEL: '{true_label_str}' (Description: {origin_description}).
However, the agent's final PREDICTION was INCORRECT: '{wrong_prediction}'.
Here is the GENERATED REPORT by the agent that led to this wrong prediction:
<generated_report>
{generated_report}
</generated_report>

TASK:
1. Analyze why the agent failed based on its generated report and current instructions.
2. Rewrite the CURRENT PROMPT (in its original language, e.g., keep it in Chinese if it's in Chinese) to explicitly address this specific failure mode so the agent does not repeat it.
3. Be careful not to degrade the prompt's general performance. Only add clarifying rules, nuanced considerations, or distinct examples related to this failure.
4. Output ONLY the complete text of the NEW PROMPT. Do not include any meta-commentary, greetings, explanations, or backticks wrapped around it. The output will be directly written into the .md file.
"""

    messages = [
        SystemMessage(content=reflection_system_prompt),
        HumanMessage(content="Please provide the updated rewritten prompt directly.")
    ]
    
    try:
        logs.info("Invoking LLM for prompt reflection...")
        response = model.invoke(messages)
        new_prompt = response.content.strip()
        
        # Simple safety check: make sure the model didn't return an empty string
        if not new_prompt:
            logs.error("Reflection generated empty prompt, aborting update.")
            return
            
        backup_file(prompt_file)
        
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(new_prompt)
            
        logs.info(f"Successfully updated prompt file: {prompt_file}")
    except Exception as e:
        logs.error(f"Error during reflection LLM invocation: {e}")
