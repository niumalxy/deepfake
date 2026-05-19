import sys
import os
import uuid
import json
import argparse
import traceback
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reflection_agent.data_loader import load_sampled_dataset
from segment_agent.graph.workflow import create_graph
from logger import logs


POSITIVE_LABEL_STR = "fake"


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


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def evaluate(
    num_per_class: int = 100,
    seed: int = 42,
    split: str = "validation",
    output_path: str = None,
) -> Dict[str, Any]:
    logs.info(f"===== Validation: split={split}, num_per_class={num_per_class}, seed={seed} =====")
    samples = load_sampled_dataset(num_per_class=num_per_class, split=split, seed=seed)
    total = len(samples)
    logs.info(f"Loaded {total} validation samples.")

    tp = fp = fn = tn = 0
    errors = 0
    per_sample: List[Dict[str, Any]] = []

    for i, (img, label_tuple, sample_desc) in enumerate(samples):
        task_id = f"val_{uuid.uuid4().hex[:8]}"
        label_int = label_tuple[0]
        label_desc = label_tuple[1]
        true_label_str = POSITIVE_LABEL_STR if label_int == 0 else "normal"

        logs.info(f"[{i + 1}/{total}] Detecting sample (label={label_desc})...")

        try:
            prediction, _ = _run_single_detection(img, label_tuple, task_id)
        except Exception:
            errors += 1
            logs.error(f"Detection crashed for {task_id}: {traceback.format_exc()}")
            per_sample.append({
                "task_id": task_id,
                "true_label": true_label_str,
                "predicted": None,
                "error": True,
                "description": sample_desc[:300],
            })
            continue

        predicted_fake = (prediction == POSITIVE_LABEL_STR)
        is_fake = (label_int == 0)

        if predicted_fake and is_fake:
            tp += 1
        elif predicted_fake and not is_fake:
            fp += 1
        elif not predicted_fake and is_fake:
            fn += 1
        else:
            tn += 1

        per_sample.append({
            "task_id": task_id,
            "true_label": true_label_str,
            "predicted": prediction,
            "correct": predicted_fake == is_fake,
            "description": sample_desc[:300],
        })

        if (i + 1) % 20 == 0:
            interim = _compute_metrics(tp, fp, fn, tn)
            logs.info(
                f"  Progress {i + 1}/{total} — "
                f"acc={interim['accuracy']:.3f} "
                f"prec={interim['precision']:.3f} "
                f"rec={interim['recall']:.3f}"
            )

    metrics = _compute_metrics(tp, fp, fn, tn)
    summary = {
        "split": split,
        "num_per_class": num_per_class,
        "seed": seed,
        "total_samples": total,
        "evaluated": tp + fp + fn + tn,
        "errors": errors,
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "metrics": metrics,
    }

    logs.info("===== Validation Complete =====")
    logs.info(f"Total: {total}, evaluated: {tp + fp + fn + tn}, errors: {errors}")
    logs.info(f"Confusion matrix — TP={tp} FP={fp} FN={fn} TN={tn}")
    logs.info(
        f"Accuracy={metrics['accuracy']:.4f}  "
        f"Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}"
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": per_sample}, f, ensure_ascii=False, indent=2)
        logs.info(f"Detailed results saved to {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation: accuracy / precision / recall on the SID validation split.")
    parser.add_argument("--num_per_class", type=int, default=100, help="Samples per class (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to dump per-sample JSON results")
    args = parser.parse_args()

    evaluate(
        num_per_class=args.num_per_class,
        seed=args.seed,
        split=args.split,
        output_path=args.output,
    )
