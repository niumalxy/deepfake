import re
from logger import logs


_VERDICT_VALUES = {"Real", "Likely Real", "Uncertain", "Likely Fake", "Fake"}


def _count_bullets(block: str) -> int:
    """计数 evidence 块中的 bullet 数；'none' 计为 0。"""
    if not block:
        return 0
    text = block.strip()
    if not text or text.lower() == "none":
        return 0
    # 按行计算以 -, *, + 或数字开头的 bullet
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[-*+]\s+\S", stripped) or re.match(r"^\d+[\.\)]\s+\S", stripped):
            count += 1
    # 若 LLM 没用 bullet 但写了实际内容，至少计为 1
    if count == 0 and text.lower() != "none":
        return 1
    return count


def _parse_region_verdict(text: str) -> dict:
    """从 LLM 输出中解析 <region_verdict> 块。缺失时回退到 Uncertain。"""
    fallback = {
        "verdict": "Uncertain",
        "confidence": 50,
        "strong_count": 0,
        "weak_count": 1,
    }
    if not text:
        return fallback

    block_match = re.search(r"<region_verdict>(.*?)</region_verdict>", text, re.DOTALL | re.IGNORECASE)
    if not block_match:
        logs.warning("region_verdict block not found in analysis output; using Uncertain fallback.")
        return fallback

    block = block_match.group(1)

    verdict_match = re.search(r"<verdict>(.*?)</verdict>", block, re.DOTALL | re.IGNORECASE)
    verdict = verdict_match.group(1).strip() if verdict_match else "Uncertain"
    if verdict not in _VERDICT_VALUES:
        for candidate in _VERDICT_VALUES:
            if candidate.lower() == verdict.lower():
                verdict = candidate
                break
        else:
            verdict = "Uncertain"

    conf_match = re.search(r"<confidence>(.*?)</confidence>", block, re.DOTALL | re.IGNORECASE)
    try:
        confidence = int(re.search(r"\d+", conf_match.group(1)).group()) if conf_match else 50
    except (AttributeError, ValueError):
        confidence = 50
    confidence = max(0, min(100, confidence))

    strong_match = re.search(r"<strong_evidence>(.*?)</strong_evidence>", block, re.DOTALL | re.IGNORECASE)
    weak_match = re.search(r"<weak_evidence>(.*?)</weak_evidence>", block, re.DOTALL | re.IGNORECASE)

    strong_count = _count_bullets(strong_match.group(1)) if strong_match else 0
    weak_count = _count_bullets(weak_match.group(1)) if weak_match else 0

    return {
        "verdict": verdict,
        "confidence": confidence,
        "strong_count": strong_count,
        "weak_count": weak_count,
    }


def next_part(state):
    current_idx = state.get('current_img_idx', 0)
    cropped_imgs = state.get("cropped_imgs", [])

    if state.get("analysis_messages"):
        analysis_text = state["analysis_messages"][-1].content
        cropped_imgs[current_idx]["analysis_result"] = analysis_text

        parsed = _parse_region_verdict(analysis_text)
        cropped_imgs[current_idx]["verdict"] = parsed["verdict"]
        cropped_imgs[current_idx]["confidence"] = parsed["confidence"]
        cropped_imgs[current_idx]["strong_count"] = parsed["strong_count"]
        cropped_imgs[current_idx]["weak_count"] = parsed["weak_count"]

        logs.info(
            f"Region {current_idx} verdict: {parsed['verdict']} "
            f"(conf={parsed['confidence']}, strong={parsed['strong_count']}, weak={parsed['weak_count']})"
        )

    cropped_imgs[current_idx]["is_done"] = True

    return {
        "cropped_imgs": cropped_imgs,
        "current_img_idx": current_idx + 1,
        "analysis_messages": [],  # 清空消息历史，结构化 verdict 已保留到 cropped_imgs
        "analysis_iter_count": 0,
        "tool_call_times": 0,
    }
