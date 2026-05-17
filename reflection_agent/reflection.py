from chat_model.openai.langchain_model import model
from langchain_core.messages import HumanMessage, SystemMessage
from entity.dump_type import DumpType
from logger import logs
import os
import shutil
import time
from typing import List, Dict, Any

PROMPT_FILE = "segment_agent/docs/constitution.md"


def backup_file(filepath: str) -> str:
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


def _build_bad_cases_text(bad_cases: List[Dict[str, Any]]) -> str:
    text_parts = []
    for i, case in enumerate(bad_cases):
        text_parts.append(
            f"CASE {i + 1}:\n"
            f"- True Label: {case['true_label']}\n"
            f"- Predicted: {case['predicted']}\n"
            f"- Image Description: {case.get('description', 'N/A')}\n"
            f"- Key Analysis Report Snippet:\n  {case.get('report_snippet', 'N/A')}"
        )
    return "\n\n".join(text_parts)


def reflect_on_bad_cases(bad_cases: List[Dict[str, Any]], prompt_file: str = None):
    """
    基于一批检测失败案例，通过原则提炼方式更新 img_part_analysis 的 prompt。

    与单案例全量重写不同，本方法：
    1. 让 LLM 分析所有 bad case，提取 1-3 条高层次原则
    2. 将这些原则精简地融入现有 prompt
    3. 控制 prompt 总长度，删除低价值或重复内容

    Args:
        bad_cases: 失败案例列表，每项含 true_label, predicted, description, report_snippet
        prompt_file: 要更新的 prompt 文件路径，默认 constitution.md
    """
    if not bad_cases:
        logs.info("No bad cases to reflect on, skipping.")
        return None

    target_file = prompt_file or PROMPT_FILE

    if not os.path.exists(target_file):
        logs.error(f"Prompt file not found: {target_file}")
        return None

    with open(target_file, "r", encoding="utf-8") as f:
        current_prompt = f.read()

    bad_cases_text = _build_bad_cases_text(bad_cases)

    system_prompt = f"""You are an Expert Prompt Engineer specializing in optimizing AI agent instructions.

Your task: improve the prompt of a deepfake detection agent that made {len(bad_cases)} classification errors.

<current_prompt>
{current_prompt}
</current_prompt>

Below are the detection failures:
<bad_cases>
{bad_cases_text}
</bad_cases>

INSTRUCTIONS:

1. Analyze all bad cases together. Identify the COMMON FAILURE PATTERNS — not individual image quirks, but recurring reasoning gaps in the agent's approach.

2. Derive 1-3 concise, high-level PRINCIPLES that, if present in the prompt, would most likely have prevented these errors. Each principle should be:
   - 1-2 sentences
   - Focused on reasoning guidance, not rigid rules
   - General enough to cover multiple failure modes

3. Integrate these principles into the prompt. Merge with existing content where overlap exists. If existing content is outdated or less valuable, replace it.

4. CRITICAL: The output prompt MUST NOT grow significantly in length. Preserve the prompt's original language (Chinese or English). Remove or condense lower-value sections if needed to make room.

5. Output ONLY the complete text of the UPDATED PROMPT. No meta-commentary, no markdown fences, no explanations.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please output the updated prompt directly, no additional text.")
    ]

    try:
        logs.info(f"Invoking LLM for principle-based reflection on {len(bad_cases)} bad cases...")
        response = model.invoke(messages)
        new_prompt = response.content.strip()

        if not new_prompt or len(new_prompt) < 100:
            logs.error("Reflection generated too-short prompt, aborting update.")
            return None

        backup_file(target_file)

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(new_prompt)

        logs.info(f"Successfully updated prompt via principle extraction: {target_file}")
        return new_prompt

    except Exception as e:
        logs.error(f"Error during principle-based reflection: {e}")
        return None


def reflect_prompt(task_id: str, dump_type: str):
    """
    原有的单案例反射函数（MQ 触发），保持向后兼容。
    现在内部也改用原则提炼方式。
    """
    from db.mongodb import get_analysis_by_task_id

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
    elif dump_type == DumpType.REPORT.value or dump_type == DumpType.REPORT:
        prompt_file = PROMPT_FILE
    else:
        logs.error(f"Unknown dump type for reflection: {dump_type}")
        return

    bad_case = {
        "true_label": true_label_str,
        "predicted": wrong_prediction,
        "description": origin_description,
        "report_snippet": generated_report[:2000] if generated_report else "N/A",
    }

    reflect_on_bad_cases([bad_case], prompt_file=prompt_file)
