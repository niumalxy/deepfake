import sys
import os
import uuid
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reflection_agent.data_loader import load_sampled_dataset
from reflection_agent.reflection import (
    reflect_on_bad_cases,
    PROMPT_FILE,
    SEGMENT_PROMPT_FILE,
    _route_target_file,
)
from segment_agent.graph.workflow import create_graph
from logger import logs
from typing import List, Dict, Any


BATCH_SIZE_DEFAULT = 10


def _run_single_detection(img, label_tuple, task_id):
    graph = create_graph(
        task_id=task_id,
        img=img,
        use_chinese=True,
        label=label_tuple,
        skip_dump=True,
    )
    initial_state = {
        "status": None,
        "content_messages": [],
        "analysis_messages": [],
        "origin_img": img,
        "cropped_imgs": [],
        "cropping_imgs": [],
        "log_id": task_id,
        "current_analysis_idx": 0,
    }
    result = graph.invoke(initial_state)

    prediction = result.get("prediction", "unknown")
    return prediction, result


def _is_prediction_correct(prediction, label_int):
    if prediction is None:
        return True
    predicted_fake = (prediction == "fake")
    is_fake = (label_int == 0)
    return predicted_fake == is_fake


def _build_bad_case(label_int, label_desc, sample_desc, prediction, result) -> Dict[str, Any]:
    """从检测结果构造一个 bad case dict，携带后续路由所需的信号。"""
    true_label_str = "fake" if label_int == 0 else "normal"
    report = result.get("report", "")
    cropping_imgs = result.get("cropping_imgs", []) or []
    cropped_imgs = result.get("cropped_imgs", []) or []
    n_screened = len(cropping_imgs) or len(cropped_imgs)
    anomaly_types = sorted({c.get("anomaly_type", "other") for c in cropping_imgs}) if cropping_imgs else []

    return {
        "true_label": true_label_str,
        "predicted": prediction or "unknown",
        "description": f"{label_desc} | {sample_desc[:300]}",
        "report_snippet": report[:2000] if report else "N/A",
        "n_screened": n_screened,
        "anomaly_types": ",".join(anomaly_types) if anomaly_types else "none",
    }


def _collect_bad_cases(num_per_class: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """运行检测、收集 bad cases，但**不**在循环内触发反思（批量化在 main_batch 里做）。"""
    logs.info(f"Loading {num_per_class} samples per class...")
    samples = load_sampled_dataset(num_per_class=num_per_class, seed=seed)

    bad_cases = []
    total = len(samples)
    correct = 0

    for i, (img, label_tuple, sample_desc) in enumerate(samples):
        task_id = f"ref_{uuid.uuid4().hex[:8]}"
        label_int = label_tuple[0]
        label_desc = label_tuple[1]

        logs.info(f"[{i + 1}/{total}] Detecting sample (label={label_desc})...")

        try:
            prediction, result = _run_single_detection(img, label_tuple, task_id)
        except Exception:
            logs.error(f"Detection crashed for {task_id}: {traceback.format_exc()}")
            continue

        if result is None:
            continue

        if _is_prediction_correct(prediction, label_int):
            correct += 1
        else:
            bad_case = _build_bad_case(label_int, label_desc, sample_desc, prediction, result)
            bad_cases.append(bad_case)
            logs.info(f"  >>> BAD CASE: true={bad_case['true_label']}, predicted={bad_case['predicted']}, "
                      f"n_screened={bad_case['n_screened']}, anomaly_types={bad_case['anomaly_types']}")

        if (i + 1) % 20 == 0:
            n = i + 1
            logs.info(f"  Progress: {n}/{total}, accuracy: {correct}/{n} = {100 * correct / n:.1f}%")

    accuracy = 100 * correct / total if total > 0 else 0
    logs.info(f"===== Detection Complete: {correct}/{total} correct ({accuracy:.1f}%), "
              f"{len(bad_cases)} bad cases collected =====")
    return bad_cases


def _group_bad_cases_by_target(bad_cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按要修改的目标 prompt 文件把 bad cases 分组。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {SEGMENT_PROMPT_FILE: [], PROMPT_FILE: []}
    for bc in bad_cases:
        target = _route_target_file(bc)
        grouped.setdefault(target, []).append(bc)
    return grouped


def _chunked(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main_batch(num_per_class: int = 100, seed: int = 42, batch_size: int = BATCH_SIZE_DEFAULT):
    """
    批量检测 → 按目标文件分组 → 按 batch_size 分块 → 每块一次反思（带长度 / 矛盾 / 验证三道闸）。

    注意：reflection.REFLECTION_ENABLED 默认 False。开关到 True 后，反思才会真正写入 prompt。
    """
    logs.info("===== Reflection Agent: Batch Detection + Patch-Based Reflection =====")

    bad_cases = _collect_bad_cases(num_per_class=num_per_class, seed=seed)
    if not bad_cases:
        logs.info("No bad cases found; prompts appear stable. Done.")
        return

    grouped = _group_bad_cases_by_target(bad_cases)
    for target_file, cases in grouped.items():
        if not cases:
            continue
        logs.info(f"--- Reflecting on {len(cases)} bad cases targeting {target_file} ---")
        chunks = list(_chunked(cases, batch_size))
        for ci, chunk in enumerate(chunks):
            if len(chunk) < batch_size and ci == len(chunks) - 1 and len(chunks) > 1:
                # 最后一块不满 batch_size 跳过，避免在小样本上抖动 prompt
                logs.info(f"  Skipping tail chunk of {len(chunk)} (< batch_size={batch_size}).")
                continue
            logs.info(f"  Chunk {ci + 1}/{len(chunks)}: reflecting on {len(chunk)} cases.")
            reflect_on_bad_cases(chunk, prompt_file=target_file)

    logs.info("===== Reflection Agent: Done =====")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reflection Agent — Batch Detection & Patch-Based Prompt Optimization")
    parser.add_argument("--num_per_class", type=int, default=100, help="Samples per class (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT,
                        help=f"Bad cases per reflection batch (default: {BATCH_SIZE_DEFAULT})")
    args = parser.parse_args()

    main_batch(num_per_class=args.num_per_class, seed=args.seed, batch_size=args.batch_size)
