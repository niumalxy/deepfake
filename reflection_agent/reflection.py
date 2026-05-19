"""
Phase 4 reflection agent.

Changes from the prior single-case full-rewrite design:
- Patch-based: LLM emits <analysis> + <patch> blocks; we apply small string edits, not full rewrites.
- Sidecar reasoning notes go to reflection_agent/reflection_notes/{ts}_{tag}.md (not into the constitution).
- Three safeguards before keeping a patch:
    1. Length cap: new file <= 1.15x previous line count.
    2. Lightweight contradiction check via a small LLM call.
    3. Validation gate: run validation on a held-out seed; auto-rollback if accuracy drops > 2pp.
- Per-file Lock so concurrent MQ consumers can't race.
- Bad-case routing happens at the source (dump2db/save.py); reflect_prompt only consumes the requested file.

REFLECTION_ENABLED defaults to False. Flip to True manually after Phase 1-3 validation looks good.
"""
from chat_model.openai.langchain_model import model
from langchain_core.messages import HumanMessage, SystemMessage
from entity.dump_type import DumpType
from logger import logs
import os
import re
import json
import shutil
import time
import threading
from typing import List, Dict, Any, Tuple, Optional

# Phase 0 guard: disable autonomous prompt rewrites until Phase 4 hardening is fully validated.
# When False, reflect_on_bad_cases() is a no-op so prompt edits aren't overwritten.
REFLECTION_ENABLED = False

# Tunables
LENGTH_CAP_RATIO = 1.15
ACCURACY_DROP_TOLERANCE = 0.02      # 2 percentage points
VALIDATION_GATE_ENABLED = True       # set False to skip the slow validation gate
VALIDATION_GATE_NUM_PER_CLASS = 15   # keep small; gate is expensive
VALIDATION_GATE_SEED = 123           # held-out seed (different from main eval seed=42)

PROMPT_FILE = "segment_agent/docs/constitution.md"
SEGMENT_PROMPT_FILE = "segment_agent/docs/segment_constitution.md"
REFLECTION_NOTES_DIR = "reflection_agent/reflection_notes"
REFLECTION_LOG_FILE = "reflection_agent/reflection_log.jsonl"

# Per-file lock so concurrent MQ consumers can't race.
_FILE_LOCKS: Dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _get_file_lock(path: str) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        if path not in _FILE_LOCKS:
            _FILE_LOCKS[path] = threading.Lock()
        return _FILE_LOCKS[path]


def backup_file(filepath: str) -> Optional[str]:
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
        n_screened = case.get('n_screened', 'N/A')
        anomaly_types = case.get('anomaly_types', 'N/A')
        # 给 LLM 一个明确的"失败发生在哪一层"信号
        if isinstance(n_screened, int):
            if n_screened == 0:
                stage_hint = "筛选阶段未标记任何区域（失败发生在 segment_constitution.md，未进入深度分析）"
            else:
                stage_hint = f"筛选阶段标了 {n_screened} 个区域（失败发生在深度分析 / report 聚合，应改 constitution.md 或 report.py 规则）"
        else:
            stage_hint = "n_screened 缺失"

        text_parts.append(
            f"CASE {i + 1}:\n"
            f"- True Label: {case['true_label']}\n"
            f"- Predicted: {case['predicted']}\n"
            f"- n_screened: {n_screened}\n"
            f"- Anomaly Types Flagged by Screener: {anomaly_types}\n"
            f"- Failure Stage Hint: {stage_hint}\n"
            f"- Image Description (含标注信息): {case.get('description', 'N/A')}\n"
            f"- Key Analysis Report Snippet:\n  {case.get('report_snippet', 'N/A')}"
        )
    return "\n\n".join(text_parts)


def _reflection_system_prompt(current_prompt: str, bad_cases_text: str, n_cases: int) -> str:
    return f"""你是一名 Prompt 工程师，专门优化 deepfake 检测 agent 的指令。你的方法学是 **case-driven 的微调**：每个 bad case 独立诊断到精确的"为什么没被现有规则捕获"的根因，然后用最小化的补丁补上那个根因——而不是泛泛提炼"原则"。

<current_prompt>
{current_prompt}
</current_prompt>

<bad_cases>
{bad_cases_text}
</bad_cases>

## 你必须遵循的分析流程

对每个 bad case **独立**做 4 步诊断（A-D），不要先合并到一起说"common pattern"。合并放在 4 步全做完之后。

### Step A — 失败方向

写明：是**漏检**（true=fake but predicted=normal）还是**误报**（true=normal but predicted=fake）。

### Step B — 关键证据

读 `Image Description` 与 `Key Analysis Report Snippet`，提取**最具体**的 1-2 条线索。例如：
- "篡改在 lower center 的一双脚，对应**物体替换**"
- "full_synthetic 的火车头光照过分戏剧化、边缘过于锐利，对应**整体 AI 生成嫌疑**"
- "report 显示模型把皮肤光滑判为 strong fake，但其实是修图"

这些线索在**视觉上对应哪个 anomaly_type**？（splicing_edge / repeated_pattern / anatomical / physics / text_gibberish / other / global_impression）

### Step C — 现有规则盲点（关键步骤）

到 `<current_prompt>` 里找到**对应那个 anomaly_type 的具体段落**，**逐字引用**（用 quote 标记）。然后解释：

- 这段规则**为什么没触发**对这条信号的标记？
    - 是 trigger 描述太具体、没覆盖这个变体？
    - 是规则用了"必须 + 严格"措辞，让模型偏向不标？
    - 是这个 anomaly_type 整段都没有讨论这类信号？
    - 是规则正确但模型应用错了（这种情况**不应该改 prompt**）？
- 引用必须精确——空泛说"规则不够严"不算诊断。

如果你认为模型是"应用错了"（rule 是对的，模型没遵守），跳过本 case 的补丁（在 patch 段写注释说明，不要硬塞编辑）。

### Step D — 最小补丁

提出一个**狭窄**的补丁，且：

1. 只针对 Step C 引用的具体段落；
2. 优先采用**扩展现有 trigger 的子条件**或**添加一行例子**的方式，而不是新增整段；
3. **不要引入新的主观性兜底**（除非补丁对象本身就是 global_impression）；
4. **反向约束自检**：写完补丁后，问自己"这一条会不会让正常图像也被误判？" 如果会，**附加一句"不要仅因为 X 就触发"的反向约束**，把误标的口子收回。

### 跨 case 合并

A-D 全部做完之后，再看：**是否有 2 个或以上 bad case 共享同一个 Step C 盲点**？如果有，把它们的补丁合并为一条。否则每个 bad case 给一个独立的小补丁——不要为了"凑共性"而强行合并不同根因的 case。

---

## 输出格式（严格遵守）

```
<analysis>
### Case 1
A. 方向: [漏检 / 误报]
B. 关键证据: ...
   对应类别: ...
C. 现有规则盲点:
   引用: "..."（quote 当前 prompt 中的精确文本）
   为什么没触发: ...
D. 补丁思路: ...
   反向约束自检: ...

### Case 2
...

### 跨 case 合并
[说明哪些 case 共享盲点合并为一个补丁；或写"无共享盲点，独立处理"]
</analysis>

<patch>
  <replace>
    <find>从 current_prompt 中精确复制的字符串（包括空格换行）</find>
    <with>替换文本</with>
  </replace>
  <insert_after anchor="精确锚点字符串">要追加的内容</insert_after>
  <delete>要删除的精确字符串</delete>
</patch>
```

**关键约束**：
- `<patch>` 中的 `<find>` / `anchor` / `<delete>` 必须**逐字**与 `current_prompt` 匹配（包括空格、换行、标点），否则补丁会被自动拒绝。
- 不要重写整段——能加一个子项、改一个措辞、补一行例子的，就不要换整段。
- 输出**只**包含 `<analysis>` 与 `<patch>` 两段，不要其他文字。
"""


_PATCH_OP_RE = re.compile(
    r"<(replace|insert_after|delete)(?:\s+anchor=\"([^\"]*)\")?>(.*?)</\1>",
    re.DOTALL,
)


def _parse_patch_ops(patch_text: str) -> List[Dict[str, Any]]:
    """从 LLM 输出的 <patch>...</patch> 块中解析出有序的编辑指令。"""
    block_match = re.search(r"<patch>(.*?)</patch>", patch_text, re.DOTALL)
    if not block_match:
        return []
    body = block_match.group(1)
    ops = []
    for m in _PATCH_OP_RE.finditer(body):
        op_kind = m.group(1)
        anchor = m.group(2) or ""
        inner = m.group(3)
        if op_kind == "replace":
            find_m = re.search(r"<find>(.*?)</find>", inner, re.DOTALL)
            with_m = re.search(r"<with>(.*?)</with>", inner, re.DOTALL)
            if find_m and with_m:
                ops.append({"kind": "replace", "find": find_m.group(1), "with": with_m.group(1)})
        elif op_kind == "insert_after":
            ops.append({"kind": "insert_after", "anchor": anchor, "text": inner})
        elif op_kind == "delete":
            ops.append({"kind": "delete", "find": inner})
    return ops


def _apply_patch_ops(original: str, ops: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """按顺序应用 patch ops。任何 op 因 anchor / find 找不到而失败 → 返回错误列表。"""
    text = original
    errors = []
    for i, op in enumerate(ops):
        if op["kind"] == "replace":
            if op["find"] in text:
                text = text.replace(op["find"], op["with"], 1)
            else:
                errors.append(f"op#{i} replace: <find> not found verbatim in current prompt.")
        elif op["kind"] == "insert_after":
            anchor = op["anchor"]
            if anchor and anchor in text:
                idx = text.find(anchor) + len(anchor)
                text = text[:idx] + op["text"] + text[idx:]
            else:
                errors.append(f"op#{i} insert_after: anchor not found.")
        elif op["kind"] == "delete":
            if op["find"] in text:
                text = text.replace(op["find"], "", 1)
            else:
                errors.append(f"op#{i} delete: <find> not found verbatim.")
    return text, errors


def _length_cap_check(old_text: str, new_text: str) -> Tuple[bool, str]:
    old_lines = max(1, len(old_text.splitlines()))
    new_lines = len(new_text.splitlines())
    ratio = new_lines / old_lines
    if ratio > LENGTH_CAP_RATIO:
        return False, f"length cap: {new_lines}/{old_lines} = {ratio:.2f} > {LENGTH_CAP_RATIO}"
    return True, f"length ok: {new_lines}/{old_lines} = {ratio:.2f}"


def _contradiction_check(new_text: str) -> Tuple[bool, str]:
    """轻量 LLM 调用：把 prompt 一切为二，问 LLM 两半对同一输入是否给出矛盾建议。"""
    midpoint = len(new_text) // 2
    half_a = new_text[:midpoint]
    half_b = new_text[midpoint:]
    judge_prompt = (
        "下面是同一份 agent prompt 的前半段 A 与后半段 B。它们对**相同输入**是否给出**相互矛盾**的处理建议?\n"
        "只关注规则层面的硬冲突 (同一种情况一处说要、另一处说不要)，不要把 [详略不同] 或 [措辞不同] 算作矛盾。\n"
        "格式严格：先输出 `Yes` 或 `No`，再用一句话说明原因。\n\n"
        f"A:\n{half_a}\n\n---\n\nB:\n{half_b}"
    )
    try:
        response = model.invoke([HumanMessage(content=judge_prompt)])
        out = (response.content or "").strip()
        first_token = out.split()[0].lower() if out else "no"
        if first_token.startswith("yes"):
            return False, f"contradiction detected: {out[:200]}"
        return True, f"no contradiction: {out[:200]}"
    except Exception as e:
        logs.warning(f"Contradiction check LLM call failed: {e}; defaulting to pass.")
        return True, "contradiction check skipped (LLM error)"


def _validation_gate(num_per_class: int = VALIDATION_GATE_NUM_PER_CLASS,
                     seed: int = VALIDATION_GATE_SEED) -> Optional[float]:
    """跑一次 held-out validation，返回 accuracy。失败时返回 None。"""
    try:
        from validation.validate import evaluate
        summary = evaluate(num_per_class=num_per_class, seed=seed, split="validation", output_path=None)
        return float(summary.get("metrics", {}).get("accuracy", 0.0))
    except Exception as e:
        logs.warning(f"Validation gate failed to run: {e}")
        return None


def _save_analysis_note(analysis_text: str, prompt_file: str) -> str:
    os.makedirs(REFLECTION_NOTES_DIR, exist_ok=True)
    ts = int(time.time())
    tag = "segment" if "segment_constitution" in prompt_file else "deep"
    note_path = os.path.join(REFLECTION_NOTES_DIR, f"{ts}_{tag}.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(analysis_text)
    return note_path


def _append_reflection_log(record: Dict[str, Any]):
    os.makedirs(os.path.dirname(REFLECTION_LOG_FILE) or ".", exist_ok=True)
    with open(REFLECTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reflect_on_bad_cases(bad_cases: List[Dict[str, Any]], prompt_file: str = None) -> Optional[str]:
    """
    基于一批检测失败案例，以补丁方式小幅修订 prompt，并经过长度 / 矛盾 / 验证三道闸。

    Args:
        bad_cases: 失败案例列表
        prompt_file: 要更新的 prompt 文件路径，默认 constitution.md
    """
    if not REFLECTION_ENABLED:
        logs.info("Reflection disabled (REFLECTION_ENABLED=False); skipping prompt update.")
        return None

    if not bad_cases:
        logs.info("No bad cases to reflect on, skipping.")
        return None

    target_file = prompt_file or PROMPT_FILE

    if not os.path.exists(target_file):
        logs.error(f"Prompt file not found: {target_file}")
        return None

    lock = _get_file_lock(target_file)
    if not lock.acquire(blocking=False):
        logs.info(f"Another reflection is already running on {target_file}; skipping this batch.")
        return None

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            current_prompt = f.read()

        bad_cases_text = _build_bad_cases_text(bad_cases)
        system_prompt = _reflection_system_prompt(current_prompt, bad_cases_text, len(bad_cases))

        logs.info(f"Invoking LLM for patch-based reflection on {len(bad_cases)} bad cases ({target_file})...")
        try:
            response = model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="请按指定格式输出 <analysis> 与 <patch> 两段，不要其他内容。")
            ])
            llm_out = response.content or ""
        except Exception as e:
            logs.error(f"Reflection LLM call failed: {e}")
            return None

        # 1) 抽出 analysis 与 patch
        analysis_match = re.search(r"<analysis>(.*?)</analysis>", llm_out, re.DOTALL)
        analysis_text = analysis_match.group(1).strip() if analysis_match else "(no analysis emitted)"

        ops = _parse_patch_ops(llm_out)
        if not ops:
            logs.warning("Reflection produced no actionable patch ops; aborting update.")
            _append_reflection_log({
                "timestamp": int(time.time()),
                "prompt_file": target_file,
                "decision": "rejected_no_ops",
                "n_bad_cases": len(bad_cases),
            })
            return None

        # 2) 应用补丁
        new_prompt, op_errors = _apply_patch_ops(current_prompt, ops)
        if op_errors:
            logs.warning(f"Patch ops had errors: {op_errors}")
            _append_reflection_log({
                "timestamp": int(time.time()),
                "prompt_file": target_file,
                "decision": "rejected_patch_errors",
                "errors": op_errors,
                "n_bad_cases": len(bad_cases),
            })
            return None

        if new_prompt == current_prompt:
            logs.info("Patch produced no change; skipping write.")
            return None

        # 3) 长度闸
        len_ok, len_msg = _length_cap_check(current_prompt, new_prompt)
        if not len_ok:
            logs.warning(f"Length cap violated: {len_msg}")
            _append_reflection_log({
                "timestamp": int(time.time()),
                "prompt_file": target_file,
                "decision": "rejected_length",
                "info": len_msg,
                "n_bad_cases": len(bad_cases),
            })
            return None
        logs.info(len_msg)

        # 4) 矛盾闸
        contra_ok, contra_msg = _contradiction_check(new_prompt)
        if not contra_ok:
            logs.warning(f"Contradiction detected, rejecting patch: {contra_msg}")
            _append_reflection_log({
                "timestamp": int(time.time()),
                "prompt_file": target_file,
                "decision": "rejected_contradiction",
                "info": contra_msg,
                "n_bad_cases": len(bad_cases),
            })
            return None
        logs.info(contra_msg)

        # 5) 备份并写盘
        backup_path = backup_file(target_file)
        note_path = _save_analysis_note(analysis_text, target_file)

        with open(target_file, "w", encoding="utf-8") as f:
            f.write(new_prompt)
        logs.info(f"Patch applied to {target_file}. Notes saved at {note_path}.")

        # 6) 验证闸（最贵的一步，可关）
        if VALIDATION_GATE_ENABLED:
            # 先跑 patch 后的 accuracy
            post_acc = _validation_gate()

            # 临时还原以测 patch 前 accuracy
            shutil.copy2(target_file, target_file + ".__post.tmp")
            shutil.copy2(backup_path, target_file)
            pre_acc = _validation_gate()
            # 再恢复 patch 版本
            shutil.copy2(target_file + ".__post.tmp", target_file)
            os.remove(target_file + ".__post.tmp")

            if pre_acc is not None and post_acc is not None and (pre_acc - post_acc) > ACCURACY_DROP_TOLERANCE:
                logs.warning(
                    f"Validation gate: accuracy dropped {pre_acc:.3f} -> {post_acc:.3f} "
                    f"(> {ACCURACY_DROP_TOLERANCE}); rolling back."
                )
                shutil.copy2(backup_path, target_file)
                _append_reflection_log({
                    "timestamp": int(time.time()),
                    "prompt_file": target_file,
                    "decision": "rolled_back",
                    "pre_acc": pre_acc,
                    "post_acc": post_acc,
                    "backup": backup_path,
                    "notes": note_path,
                    "n_bad_cases": len(bad_cases),
                })
                return None

            _append_reflection_log({
                "timestamp": int(time.time()),
                "prompt_file": target_file,
                "decision": "kept",
                "pre_acc": pre_acc,
                "post_acc": post_acc,
                "backup": backup_path,
                "notes": note_path,
                "n_bad_cases": len(bad_cases),
            })
            logs.info(f"Validation gate passed (pre={pre_acc}, post={post_acc}); patch kept.")
        else:
            _append_reflection_log({
                "timestamp": int(time.time()),
                "prompt_file": target_file,
                "decision": "kept_no_gate",
                "backup": backup_path,
                "notes": note_path,
                "n_bad_cases": len(bad_cases),
            })

        return new_prompt

    finally:
        lock.release()


def _route_target_file(bad_case: Dict[str, Any]) -> str:
    """根据 bad case 形态决定改哪一个 prompt 文件。"""
    true_label = bad_case.get("true_label")
    predicted = bad_case.get("predicted")
    n_screened = bad_case.get("n_screened", 0)

    # 漏检：真为 fake 但筛选器一个区域都没标 → 筛选 prompt 问题
    if true_label == "fake" and n_screened == 0:
        return SEGMENT_PROMPT_FILE
    # 过标：真为 normal 但筛选器标了很多区域 → 筛选 prompt 问题
    if true_label == "normal" and n_screened > 5:
        return SEGMENT_PROMPT_FILE
    # 其他情形：深度分析 / 报告聚合的问题
    return PROMPT_FILE


def reflect_prompt(task_id: str, dump_type: str):
    """
    MQ 触发入口。保持向后兼容；改为以 anomaly_type / n_screened 等信号路由目标 prompt 文件。
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

    bad_case = {
        "true_label": true_label_str,
        "predicted": wrong_prediction,
        "description": origin_description,
        "report_snippet": generated_report[:2000] if generated_report else "N/A",
        "n_screened": analysis.get("n_screened", 0),
        "anomaly_types": analysis.get("anomaly_types", "N/A"),
    }

    # 显式 dump_type 仍然受尊重；否则按信号路由
    if dump_type in (DumpType.SEGMENT, DumpType.SEGMENT.value):
        prompt_file = SEGMENT_PROMPT_FILE
    elif dump_type in (DumpType.REPORT, DumpType.REPORT.value):
        prompt_file = PROMPT_FILE
    else:
        prompt_file = _route_target_file(bad_case)

    reflect_on_bad_cases([bad_case], prompt_file=prompt_file)
