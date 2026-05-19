from typing import Dict, Any
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from segment_agent.graph.state import AgentState
from entity.segment_agent_status import AgentStatus
from logger import logs
from chat_model.openai.langchain_model import model
from utils.img_convert import img_to_base64
from segment_agent.nodes.report.prompt import get_summary_system_prompt


# Phase 3 决策规则常量。需要调参时在这里改，不动 prompt。
REPORT_DECISION_RULES = {
    "strong_fake_min": 1,                  # 强证据伪造区域 ≥ 此值 → fake
    "weak_fake_per_region": 2,             # 单个区域被计为 weak_fake_region 所需的 weak evidence 数
    "global_impression_min_confidence": 70, # global_impression 兜底为 strong 所需的最低 confidence
}


def _aggregate_cropped_verdicts(cropped_imgs, image_size):
    """对所有区域的结构化 verdict 做确定性聚合。"""
    total = len(cropped_imgs)
    strong_fake_regions = 0
    weak_fake_regions = 0
    real_regions = 0
    uncertain_regions = 0
    flagged_area = 0
    confidences = []

    for c in cropped_imgs:
        verdict = c.get("verdict", "Uncertain")
        strong = c.get("strong_count", 0)
        weak = c.get("weak_count", 0)
        confidence = c.get("confidence", 0)
        anomaly_type = c.get("anomaly_type", "other")
        confidences.append(confidence)

        # 普通区域：要求 strong_count >= 1 才计入 strong_fake_regions
        normal_strong = verdict in ("Fake", "Likely Fake") and strong >= 1
        # global_impression 区域：放宽——无需 strong_count，只要 verdict 为 Likely Fake/Fake 且 confidence 达标
        gi_strong = (
            anomaly_type == "global_impression"
            and verdict in ("Fake", "Likely Fake")
            and confidence >= REPORT_DECISION_RULES["global_impression_min_confidence"]
        )

        if normal_strong or gi_strong:
            strong_fake_regions += 1
        elif verdict == "Likely Fake" and strong == 0 and weak >= REPORT_DECISION_RULES["weak_fake_per_region"]:
            weak_fake_regions += 1
        elif verdict in ("Real", "Likely Real"):
            real_regions += 1
        elif verdict == "Uncertain":
            uncertain_regions += 1

        if verdict in ("Fake", "Likely Fake"):
            tl = c.get("top_left", (0, 0))
            br = c.get("bottom_right", (0, 0))
            try:
                flagged_area += max(0, br[0] - tl[0]) * max(0, br[1] - tl[1])
            except (TypeError, IndexError):
                pass

    image_area = image_size[0] * image_size[1] if image_size else 0
    flagged_area_fraction = (flagged_area / image_area) if image_area > 0 else 0.0
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0

    return {
        "total": total,
        "strong_fake_regions": strong_fake_regions,
        "weak_fake_regions": weak_fake_regions,
        "real_regions": real_regions,
        "uncertain_regions": uncertain_regions,
        "flagged_area_fraction": flagged_area_fraction,
        "avg_confidence": avg_confidence,
    }


def _deterministic_prediction(stats):
    """基于聚合统计的硬判决规则。LLM 报告中的 prediction 仅供参考，最终值以此为准。"""
    if stats["strong_fake_regions"] >= REPORT_DECISION_RULES["strong_fake_min"]:
        return "fake"
    return "normal"


def report(state: AgentState, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成最终的分析报告节点。

    Phase 3 改造:
    - LLM 调用前先用确定性规则聚合各区域 verdict
    - 把聚合统计注入到 system prompt
    - 最终 prediction 取确定性结果，LLM 输出的 prediction 仅做 sanity check
    """
    logs.info("--- Generating Final Report with Model ---")

    cropped_imgs = state.get('cropped_imgs', [])
    origin_img = state.get('origin_img')

    image_size = origin_img.size if origin_img else None
    stats = _aggregate_cropped_verdicts(cropped_imgs, image_size)
    deterministic_pred = _deterministic_prediction(stats)

    logs.info(
        f"Aggregation: total={stats['total']}, "
        f"strong_fake={stats['strong_fake_regions']}, "
        f"weak_fake={stats['weak_fake_regions']}, "
        f"real={stats['real_regions']}, "
        f"uncertain={stats['uncertain_regions']}, "
        f"flagged_area={stats['flagged_area_fraction']:.2%}, "
        f"deterministic={deterministic_pred}"
    )

    if origin_img:
        img_base64 = img_to_base64(origin_img)
        origin_img_content = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        }]
    else:
        origin_img_content = []
        logs.warning("No origin image found for report generation")

    analysis_text = "以下是对图片局部的详细分析结果："
    for idx, msg in enumerate(cropped_imgs):
        this_analysis = {
            "location": msg.get("items", ""),
            "verdict": msg.get("verdict", "Uncertain"),
            "confidence": msg.get("confidence", 0),
            "strong_count": msg.get("strong_count", 0),
            "weak_count": msg.get("weak_count", 0),
            "analysis_result": msg.get("analysis_result", ""),
        }
        analysis_text += f"\n{this_analysis}"

    user_content = []
    if origin_img_content:
        user_content.extend(origin_img_content)
    user_content.append({
        "type": "text",
        "text": (
            f"以下是图像各部分的详细分析结果，请基于这些信息生成一份完整的分析报告：\n\n{analysis_text}\n\n"
            f"## 系统聚合统计（供你参考，已应用决策规则）\n"
            f"- 强证据伪造区域数: {stats['strong_fake_regions']}\n"
            f"- 弱证据伪造区域数: {stats['weak_fake_regions']}\n"
            f"- 真实/可能真实区域数: {stats['real_regions']}\n"
            f"- 不确定区域数: {stats['uncertain_regions']}\n"
            f"- 标记区域占图像面积比: {stats['flagged_area_fraction']:.2%}\n"
            f"- 平均置信度: {stats['avg_confidence']:.1f}\n"
            f"- 系统判决（确定性规则）: <prediction>{deterministic_pred}</prediction>\n"
        )
    })

    use_chinese = config.get('use_chinese', False)
    system_prompt = get_summary_system_prompt(use_chinese)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]

    try:
        writer = get_stream_writer()
        parts = []
        for chunk in model.stream(messages):
            delta = chunk.content if isinstance(chunk.content, str) else ""
            if delta:
                parts.append(delta)
                writer({"kind": "stream_delta", "node": "report", "delta": delta})
        generated_report = "".join(parts)
        writer({"kind": "stream_end", "node": "report"})

        logs.info("Final report generated successfully with model")
        logs.info(f"Generated report content: {generated_report[:500]}...")

        # 最终 prediction 以确定性聚合为准，不再信任 LLM 自由输出
        prediction_val = deterministic_pred

        # LLM 输出的 prediction 仅作健康检查
        llm_pred_match = re.search(r"<prediction>(.*?)</prediction>", generated_report, re.DOTALL)
        if llm_pred_match:
            llm_pred = llm_pred_match.group(1).strip().lower()
            if llm_pred != deterministic_pred:
                logs.warning(
                    f"LLM prediction '{llm_pred}' differs from deterministic '{deterministic_pred}'; "
                    f"using deterministic result."
                )
                # 把报告里的 prediction 标签替换为最终值，保持输出一致
                generated_report = re.sub(
                    r"<prediction>.*?</prediction>",
                    f"<prediction>{deterministic_pred}</prediction>",
                    generated_report,
                    count=1,
                    flags=re.DOTALL,
                )
        else:
            # LLM 没输出 prediction 标签，追加一个
            generated_report = generated_report + f"<prediction>{deterministic_pred}</prediction>"

        return {
            "report": generated_report,
            "prediction": prediction_val,
            "status": AgentStatus.FINISHED,
        }
    except Exception as e:
        logs.error(f"Error generating report with model: {e}")
        return {
            "report": "",
            "prediction": deterministic_pred,
            "status": AgentStatus.INVALID,
        }
