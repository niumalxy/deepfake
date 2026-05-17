import os
import sys
sys.path.append(".")
from logger import logging
if os.environ.get('HF_ENDPOINT', '') != "https://huggingface.co":
    logging.warning("HF_ENDPOINT not set to huggingface.co, correcting...")
    os.environ['HF_ENDPOINT'] = "https://huggingface.co"
    proxy = "http://127.0.0.1:7897"
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
from datasets import load_dataset
from typing import List, Tuple
from PIL import Image


SID_LABEL_MAP = {
    0: "real_image",
    1: "full_synthetic",
    2: "tampered",
}

INTERNAL_FAKE_LABEL = 0
INTERNAL_REAL_LABEL = 1


def _map_label(sid_label: int) -> Tuple[int, str]:
    """
    将 SID 数据集 label 映射为项目内部 label 格式 (int, str)
    - SID 0 (real) → (1, "real_image")
    - SID 1 (full_synthetic) → (0, "full_synthetic")
    - SID 2 (tampered) → (0, "tampered")
    """
    description = SID_LABEL_MAP.get(sid_label, "unknown")
    if sid_label == 0:
        return (INTERNAL_REAL_LABEL, description)
    else:
        return (INTERNAL_FAKE_LABEL, description)


def load_sampled_dataset(
    num_per_class: int = 100,
    split: str = "train",
    seed: int = 42
) -> List[Tuple[Image.Image, Tuple[int, str], str]]:
    """
    从 SID 数据集中加载样本，保持正常/伪造 1:1 比例。

    real 类加载 num_per_class 条，full_synthetic 和 tampered 各加载 num_per_class // 2 条，
    使得正常:伪造 = 1:1。

    Returns:
        List of (image, label_tuple, sample_description) tuples.
        label_tuple = (int, str) where int==0 means fake, int==1 means real.
        sample_description = 数据集自带的详细描述文本（来自 description 字段）
    """
    ds = load_dataset("saberzl/SID_Set_description", cache_dir="./reflection_agent/dataset", split=split)

    fake_num_per_class = max(num_per_class // 2, 1)
    class_configs = [
        (0, "real", num_per_class),
        (1, "full_synthetic", fake_num_per_class),
        (2, "tampered", fake_num_per_class),
    ]

    samples = []

    for sid_label, class_name, target_num in class_configs:
        subset = ds.filter(lambda x: x["label"] == sid_label)

        subset_size = len(subset)
        actual_num = min(target_num, subset_size)

        shuffled = subset.shuffle(seed=seed + sid_label)
        selected = shuffled.select(range(actual_num))

        for idx, row in enumerate(selected):
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
            if img.mode != "RGB":
                img = img.convert("RGB")

            internal_label = _map_label(sid_label)
            raw_desc = row.get("description", "")
            sample_desc = f"[{class_name}#{idx}] {raw_desc[:500]}" if raw_desc else f"{class_name}_sample_{idx}"
            samples.append((img, internal_label, sample_desc))

        logging.info(f"Loaded {actual_num} {class_name} samples (total available: {subset_size})")

    logging.info(f"Total samples loaded: {len(samples)}, shuffling to mix classes...")

    import random
    random.seed(seed)
    random.shuffle(samples)

    return samples


if __name__ == "__main__":
    samples = load_sampled_dataset(num_per_class=2)
    for img, label, desc in samples:
        print(f"Desc: {desc[:100]}..., Label: {label}, Size: {img.size}")
