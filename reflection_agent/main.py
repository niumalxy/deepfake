import sys
import os
import uuid
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reflection_agent.data_loader import load_sampled_dataset
from reflection_agent.reflection import reflect_on_bad_cases
from segment_agent.graph.workflow import create_graph
from logger import logs
from typing import List, Dict, Any


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


def _collect_bad_cases(
    num_per_class: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    logs.info(f"Loading {num_per_class} samples per class...")
    samples = load_sampled_dataset(num_per_class=num_per_class, seed=seed)

    bad_cases = []
    total = len(samples)
    correct = 0
    reflect_count = 0

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
            true_label_str = "fake" if label_int == 0 else "normal"
            report = result.get("report", "")
            bad_case = {
                "true_label": true_label_str,
                "predicted": prediction or "unknown",
                "description": f"{label_desc} | {sample_desc[:300]}",
                "report_snippet": report[:2000] if report else "N/A",
            }
            bad_cases.append(bad_case)
            logs.info(f"  >>> BAD CASE: true={true_label_str}, predicted={prediction}")

            reflect_count += 1
            logs.info(f"  >>> Reflecting on bad case #{reflect_count} to update prompt...")
            reflect_on_bad_cases([bad_case])
            logs.info(f"  >>> Reflection #{reflect_count} complete.")

        if (i + 1) % 20 == 0:
            n = i + 1
            logs.info(f"  Progress: {n}/{total}, accuracy: {correct}/{n} = {100 * correct / n:.1f}%")

    accuracy = 100 * correct / total if total > 0 else 0
    logs.info(f"===== Detection Complete: {correct}/{total} correct ({accuracy:.1f}%), {len(bad_cases)} bad cases in total, {reflect_count} reflections performed =====")
    return bad_cases


def main_batch(num_per_class: int = 100, seed: int = 42):
    """
    批量检测入口：从数据集采样三类各 N 条，运行 segment_agent，
    每发现预测失败的 bad case，立即通过原则提炼方式更新 constitution.md。
    """
    logs.info("===== Reflection Agent: Batch Detection + Incremental Prompt Optimization =====")

    bad_cases = _collect_bad_cases(num_per_class=num_per_class, seed=seed)

    if not bad_cases:
        logs.info("No bad cases found, prompt is already optimal.")
        return

    logs.info("===== Reflection Agent: Done =====")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reflection Agent - Batch Deepfake Detection & Prompt Optimization")
    parser.add_argument("--num_per_class", type=int, default=100, help="Samples per class (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    main_batch(num_per_class=args.num_per_class, seed=args.seed)
