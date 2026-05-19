# Deepfake Detection Agent — 项目现状与变更记录

> 本文档由 2026-05-18 的优化迭代产出，记录了对 segment_agent 与 reflection_agent 的系统性改造细节，并描述项目当前架构与运行方式。
> 后续每次大规模改动应继续追加到本文件的"变更记录"章节，保持时间线连续。

---

## 一、项目目标

构建一个针对 **AI 生成图像** 的检测 agent，专注于识别图像中违反**物理规律 / 自然规律 / 空间一致性**的异常。检测对象既包含整图生成的 deepfake，也包含局部编辑（splicing、inpainting、object insertion / removal 等）。

**当前主要痛点（本次优化解决的）**:
1. 大量正常图像被误判为 fake（系统性偏向 fake）
2. Reflection 学习不稳定——每个 bad case 立即触发整段 prompt 重写，导致 constitution 中出现自相矛盾的规则

---

## 二、技术栈

| 组件 | 选型 |
|---|---|
| 工作流编排 | LangGraph (StateGraph) |
| 多模态 LLM | Zhipu **GLM-4.6V** (`open.bigmodel.cn`) |
| LLM 客户端 | LangChain `ChatOpenAI` 兼容封装（`chat_model/openai/langchain_model.py`） |
| 向量检索 | FAISS IndexFlatIP + CLIP ViT-base-patch32 (512-d) |
| 持久化 | MongoDB (analysis / segment 表)、RabbitMQ (segment_queue / report_queue) |
| Web 框架 | FastAPI (`main.py`)，前端 `templates/index.html` |
| **依赖管理** | **uv**（lockfile 为 `uv.lock`；所有运行命令需 `uv run` 前缀） |
| 操作系统 | Windows 11，bash 终端 |

---

## 三、当前架构

### 3.1 端到端 segment_agent 工作流（LangGraph）

```
START
  │
  ├─ init                                  # 状态初始化
  │
  ├─ [optional] rag_node ↔ rag_tool_call   # need_rag=True 时启用，查询 FAISS 历史相似案例
  │
  ├─ img_content                            # 中性筛选可疑区域（Phase 1 重写）
  │   ├─ 输出 cropping_imgs (≤5)，含 anomaly_type 字段
  │   └─ has_suspicious_regions?
  │       ├─ NO  → workflow_end (prediction="normal")
  │       └─ YES ↓
  │
  ├─ img_cropping                           # 物理切割并保存子图
  │
  ├─ img_part_analysis (loop)               # 对每个区域跑深度分析（Phase 2 重写 prompt）
  │   ├─ 工具调用最多 1 次（view_original_image, crop, resize, ...）
  │   └─ 完成后输出 <region_verdict> XML 块
  │
  ├─ next_part                              # 解析 <region_verdict>，填入 verdict/confidence/strong_count/weak_count
  │
  ├─ report                                 # 确定性聚合 + 受约束 LLM 总结（Phase 3 改造）
  │   ├─ 计算 strong_fake_regions / weak_fake_regions / area_fraction
  │   ├─ 应用判决规则：strong_fake_regions ≥ 1 → fake, 否则 normal
  │   └─ LLM 仅负责把结构化结果翻译为可读 Markdown 报告
  │
  └─ workflow_end (dump2db)                 # 写 MongoDB + 投递 MQ + 插入 FAISS embedding
```

### 3.2 Reflection 学习循环（Phase 4 重构）

```
触发源：
  1. MQ 消费者（实时）：从 segment_queue / report_queue 拿到 task_id → reflect_prompt()
  2. 批量入口（离线）：python -m reflection_agent.main → main_batch()

reflect_on_bad_cases(batch, prompt_file)
  │
  ├─ per-file Lock（防止两个消费者同时改一个文件）
  │
  ├─ 构造系统 prompt: <current_prompt> + <bad_cases>
  │
  ├─ LLM 输出 <analysis>...</analysis><patch>...</patch>
  │
  ├─ analysis 保存到 reflection_agent/reflection_notes/{ts}_{tag}.md
  │
  ├─ 解析 patch ops: <replace>/<insert_after>/<delete>
  │
  ├─ 三道闸：
  │   1. 长度闸：new_lines / old_lines ≤ 1.15
  │   2. 矛盾闸：把新 prompt 一切为二，LLM 判 Yes/No 是否矛盾
  │   3. 验证闸：跑 validation/validate.py held-out 种子，accuracy 下降 > 2pp → 回滚
  │
  ├─ 写盘前 backup_file()，回滚时从备份恢复
  │
  └─ 决策追加到 reflection_agent/reflection_log.jsonl
```

### 3.3 关键状态字段（`segment_agent/graph/state.py`）

```python
class CroppingImg(TypedDict):
    items, top_left, bottom_right, description, save_path
    anomaly_type       # Phase 1 新增: splicing_edge | repeated_pattern | anatomical | physics | text_gibberish | other

class CroppedImg(TypedDict):
    save_path, items, description, is_done, analysis_result
    # Phase 3 新增 ↓
    verdict            # Real | Likely Real | Uncertain | Likely Fake | Fake
    confidence         # 0-100
    strong_count
    weak_count
    anomaly_type       # 从 CroppingImg 透传
    top_left, bottom_right  # 用于 report 计算 flagged_area_fraction
```

### 3.4 关键 Prompt 文件

| 文件 | 角色 | 被谁加载 | 当前状态 |
|---|---|---|---|
| `segment_agent/docs/segment_constitution.md` | 筛选 prompt（中性触发） | `img_content` 节点 | Phase 1 重写，中文 |
| `segment_agent/docs/constitution.md` | 深度分析 prompt（默认 Real + 证据评分） | `img_part_analysis` 节点 | Phase 2 重写，中文 |
| `segment_agent/nodes/report/prompt.py` | 报告系统 prompt（强制采用聚合判决） | `report` 节点 | Phase 3 改写，中文+英文双版本 |
| `segment_agent/docs/constitution_baseline_phase0.md` | Phase 0 基线快照 | 不被加载，仅供 diff | 永久保留 |
| `segment_agent/docs/segment_constitution_baseline_phase0.md` | Phase 0 基线快照 | 不被加载，仅供 diff | 永久保留 |
| `segment_agent/docs/constitution_backup_*.md` | 历史 reflection 备份（13 个） | 不被加载 | 保留作为论文写作证据 |

---

## 四、本次变更明细（按 Phase 列出）

### Phase 0：冻结 Reflection + 留存基线

**目的**：在改动任何 prompt 前，先停用自动重写，并锁定一份"修改前"基线供后续 diff。

| 文件 | 改动 |
|---|---|
| `reflection_agent/reflection.py` | 顶部新增 `REFLECTION_ENABLED = False`；`reflect_on_bad_cases()` 在 False 时 no-op |
| `segment_agent/docs/constitution_baseline_phase0.md` | **新建**：当前 constitution.md 的快照 |
| `segment_agent/docs/segment_constitution_baseline_phase0.md` | **新建**：当前 segment_constitution.md 的快照 |

---

### Phase 1：重写 segment_constitution.md（筛选 prompt）

**目的**：消除原 prompt 中"宁可错标，不可漏标"带来的高假阳率，把筛选改为**中性、客观证据触发**。

| 文件 | 改动 |
|---|---|
| `segment_agent/docs/segment_constitution.md` | **整体重写**：删除"宁可错标"；空输出 `{}` 改为"被鼓励的结果"；最多标记 5 个区域；每类异常使用"标记仅当 ... + 不要仅因为 X" 结构 |
| `segment_agent/graph/state.py` | `CroppingImg` 增加 `anomaly_type: str` 字段 |
| `segment_agent/nodes/img_content/img_content.py` | JSON 解析读取 `anomaly_type`（默认 "other"），并传入 `CroppingImg` |

**核心结构变化**:
- 旧版本 76 行 → 新版本约 100 行（结构化但更清晰）
- 新增 6 个 anomaly_type 枚举：`splicing_edge | repeated_pattern | anatomical | physics | text_gibberish | other`
- 强调"说不出具体异常就不要标记"，明确告诉模型空输出无下游成本

---

### Phase 2：重写 constitution.md（深度分析 prompt）翻译为中文 + 解决矛盾

**目的**：解决原 prompt 内部矛盾（Section 1 说"纹理光滑是弱证据"，Section 2 又说"对静物纹理不连续是强证据"），并把"默认 Real + 证据等级 rubric"提到文档最前面。

| 文件 | 改动 |
|---|---|
| `segment_agent/docs/constitution.md` | **整体重写**：从英文翻译为中文；置顶"核心立场"（默认 Real）+"证据强度评分"（强/弱/无）；明确判决规则；删除整个 "Uncanny Valley" 章节；解决静物纹理矛盾；输出 schema 改为 `<region_verdict>` XML 块 |
| `segment_agent/nodes/img_part_analysis/prompt.py` | `open(..., "r")` 改为 `open(..., "r", encoding="utf-8")`，修复 Windows 中文 latent bug |

**判决规则（新规）**:
- ≥ 1 条强证据 → Likely Fake / Fake
- 0 条强 + ≥ 2 条独立弱（不同类别）→ Uncertain（**禁止**输出 Fake）
- 其他 → Likely Real / Real

**输出 schema（新）**:
```xml
<region_verdict>
  <verdict>Real | Likely Real | Uncertain | Likely Fake | Fake</verdict>
  <confidence>0-100</confidence>
  <strong_evidence>- ...; 或 none</strong_evidence>
  <weak_evidence>- ...; 或 none</weak_evidence>
  <dismissed_artifacts>- ...; 或 none</dismissed_artifacts>
</region_verdict>
<complete>
```

---

### Phase 3：Report 节点结构化聚合 + 确定性判决

**目的**：把最终 fake/normal 决策从"LLM 自由判断"改为"先确定性聚合统计 → 再让 LLM 翻译"。

| 文件 | 改动 |
|---|---|
| `segment_agent/graph/state.py` | `CroppedImg` 新增 `verdict / confidence / strong_count / weak_count / anomaly_type / top_left / bottom_right` |
| `segment_agent/nodes/img_segment/img_segment.py` | 创建 `CroppedImg` 时透传 `anomaly_type` 与切割坐标 |
| `segment_agent/nodes/next_part/next_part.py` | **整体重写**：从最后一条 analysis message 解析 `<region_verdict>` 块；提取 verdict/confidence/strong_count/weak_count；缺失时 fallback 为 Uncertain |
| `segment_agent/nodes/report/report.py` | **整体重写**：新增 `_aggregate_cropped_verdicts()` 与 `_deterministic_prediction()`；LLM 调用前先做聚合；最终 prediction 来自聚合规则，LLM 输出仅做 sanity check（不一致时强制对齐） |
| `segment_agent/nodes/report/prompt.py` | **整体重写**：中文化；强制 LLM 采用系统聚合的 `<prediction>` 值；明确禁止仅凭整体印象输出 fake |
| `segment_agent/nodes/dump2db/save.py` | `insert_analysis()` 写入 `n_screened` 与 `anomaly_types`，供 reflection 路由使用 |

**确定性判决规则常量（`report.py` 顶部）**:
```python
REPORT_DECISION_RULES = {
    "strong_fake_min": 1,
    "weak_fake_per_region": 2,
}
```

---

### Phase 4：Reflection 改为批量 + 补丁 + 三道闸 + 自动回滚

**目的**：消除"单 bad case 全量重写"导致的 prompt 漂移；保证每次反思都有可审计的推理痕迹、容量约束、矛盾检测、与精度安全网。

| 文件 | 改动 |
|---|---|
| `reflection_agent/reflection.py` | **整体重写**：补丁式（`<analysis>` + `<patch>`）；`<replace>/<insert_after>/<delete>` 三种 op；分析推理保存到 sidecar 文件；长度闸 1.15×；矛盾闸（轻量 LLM 调用）；验证闸（跑 validate.py held-out seed）；自动回滚；per-file Lock；reflect_prompt 根据 `n_screened`/`anomaly_types` 路由到正确的 prompt 文件 |
| `reflection_agent/main.py` | **整体重写**：去掉循环内的逐条触发；改为收集所有 bad cases → 按目标 prompt 文件分组 → 按 `batch_size`（默认 10）分块反思；新增 `--batch_size` 参数 |
| `reflection_agent/reflection_notes/` | **新建目录**：sidecar 推理记录（每次反思一份 markdown） |
| `reflection_agent/reflection_log.jsonl` | **运行时新建**：每次反思的决策记录（kept / rolled_back / rejected_*） |

**可调参数（`reflection.py` 顶部）**:
```python
REFLECTION_ENABLED = False            # 默认禁用，验证完 Phase 1-3 后手工置 True
LENGTH_CAP_RATIO = 1.15
ACCURACY_DROP_TOLERANCE = 0.02
VALIDATION_GATE_ENABLED = True
VALIDATION_GATE_NUM_PER_CLASS = 15
VALIDATION_GATE_SEED = 123
```

**Bad case 路由规则（`_route_target_file`）**:
- True=fake & n_screened==0 → 漏检 → 改 `segment_constitution.md`
- True=normal & n_screened>5 → 过标 → 改 `segment_constitution.md`
- 其他 → 深度分析 / 报告问题 → 改 `constitution.md`

---

## 五、运行方式

> **重要**：项目用 **uv** 管理依赖。所有命令需 `uv run` 前缀，否则会用错 Python 环境。

### 5.1 安装依赖

```bash
uv sync
```

### 5.2 启动 Web 服务

```bash
uv run python main.py
```
- 浏览器访问 `http://localhost:8000`
- 支持两种 agent 模式：标准模式 / 局部检测模式（segment_agent）

### 5.3 单图检测（API）

```bash
curl -X POST http://localhost:8000/api/task/segment \
  -F "image=@/path/to/img.jpg" \
  -F "use_chinese=true" \
  -F "need_rag=false"
```

### 5.4 批量验证（用于对比 Phase 前后指标）

```bash
# Phase 0 基线
uv run python validation/validate.py --num_per_class 50 --seed 42 --split validation --output validation/results/phase0_baseline.json

# Phase 1-3 落地后
uv run python validation/validate.py --num_per_class 50 --seed 42 --split validation --output validation/results/phase3_report.json
```

输出包含 confusion matrix、accuracy、precision、recall。

### 5.5 启用 reflection 学习

1. **先**确认 Phase 1-3 在 validation 集上效果好于基线；
2. 把 `reflection_agent/reflection.py:30` 的 `REFLECTION_ENABLED` 改为 `True`；
3. 跑批量反思：
   ```bash
   uv run python reflection_agent/main.py --num_per_class 30 --seed 7 --batch_size 10
   ```
4. 检查 `reflection_agent/reflection_log.jsonl` 与 `reflection_agent/reflection_notes/` 验证 patch 的合理性。

如果想加速迭代，可临时把 `VALIDATION_GATE_ENABLED = False`，但失去 accuracy 安全网。

---

## 六、已知限制与风险

| 项 | 说明 |
|---|---|
| FAISS 库样本量小 | 当前 60 条历史案例，RAG 容易过拟合到特定 GAN 类型；扩库前慎用 `need_rag=True` |
| 验证门慢 | 每次 reflection 需要跑 15×2 = 30 个 detection，外加再跑一次 patch 后 validation，耗时约 5-15 分钟。可临时关闭 |
| LLM 输出结构化 verdict 的稳定性 | 若 GLM 拒绝输出 `<region_verdict>` XML，`next_part.py` 会 fallback 为 `Uncertain / 0 strong / 1 weak`，最终 report 容易倒向 normal。需要在 validation 上观察 fallback 命中率 |
| Reflection 验证种子单一 | 验证闸只用 seed=123 跑 15 张/类。如果该子集偏置，可能误判 patch 好坏。可手工把 `VALIDATION_GATE_SEED` 改成滚动种子 |
| Windows 中文 codepage | 终端默认 GBK，遇到 UTF-8 字符可能乱码显示，但不影响代码执行 |
| 13 个历史 backup 文件 | 保留在 `segment_agent/docs/` 下作为论文写作素材，不被加载；后续可统一移到 `_archive/` 子目录 |

---

## 七、变更记录（按时间）

### 2026-05-18 — Phase 0 ~ Phase 4 系统性重构

详见上文第四章。摘要：

- **Phase 0**：冻结 reflection，建立基线快照
- **Phase 1**：筛选 prompt 改为中性触发，新增 `anomaly_type` 字段
- **Phase 2**：深度分析 prompt 中文化，置顶"默认 Real + 证据评分"，解决静物纹理矛盾，新输出 schema
- **Phase 3**：Report 节点确定性聚合，最终判决脱离 LLM 自由发挥
- **Phase 4**：Reflection 改为批量 + 补丁式 + 三道闸 + 自动回滚

### 2026-05-18 — 补丁：在深度分析 prompt 中引导 `view_original_image`

**背景**：原图在第一次进入 `img_part_analysis` 时随消息注入，但 `prune_image_context(max_images=2)` 在任何一次工具调用后会把原图替换为占位文本。`view_original_image` 工具能把原图重新塞回，但 constitution.md 此前完全没提到这个工具，模型不知道何时该用。

**改动**：
- `segment_agent/docs/constitution.md`：在"推理流程"与"输出格式"之间插入新章节 **"工具使用建议"**，明确：
    - 每区域工具上限 1 次（与 `tool_call.py:44` 的硬约束对齐）；
    - `view_original_image` 应用在需要整图上下文的判断（光照 / 阴影 / 反射 / 背景直线等）或当上下文提示原图已省略时；
    - `execute_image_skill` 应用在局部数字取证操作；
    - 不调用工具也允许，避免为凑配额而无效调用。

**预期效果**：减少模型在判断光照 / 反射等需全局比对的指标时陷入"只看裁剪区域、信息不足"的盲区。

### 2026-05-19 — 迭代优化筛选 prompt + 修复 LangGraph 状态传播 bug

**背景**：跑 `validate.py --num_per_class 3` 时发现两个互相纠缠的问题：

1. **dump2db latent bug**：`save.py:27-28` 用 `state["prediction"] = "normal"` 就地修改状态。LangGraph 节点必须 **返回 dict** 才能传播更新——in-place mutation 被丢弃。在原"宁可错标"prompt 下这条捷径几乎不被走，bug 隐藏；Phase 1 中性筛选让这条路径成为主路径，bug 显形——所有"无可疑区域"图像的 prediction 一直停在 `"unknown"`。
2. **筛选 prompt 过度保守**：Phase 1 的 segment_constitution.md 在三个地方反复说"`{}` 是被鼓励的结果"，加上 global_impression 被列为带"严格规则"的兜底，模型对所有图像（包括 full_synthetic 和 tampered）一律输出 `{}`。

**改动**：

| 文件 | 内容 |
|---|---|
| `segment_agent/nodes/dump2db/save.py` | 改返回类型 `None → Dict[str, Any]`；用本地 `state_updates` 字典累积更新；在所有 return 点返回 `state_updates`；这样 LangGraph 才能把 `prediction="normal"` 写进 graph state |
| `segment_agent/docs/segment_constitution.md` | **整体重构**：删除三处"`{}` 是被鼓励的"框架；把"角色定位"从"严格守门人"改为"候选生成器"（"任何可指认局部异常 → 标记给下游审查"+"整体怀疑 AI 生成 → 用 global_impression"）；显式加"检查流程"（先扫局部 → 再评整体观感 → 都无再 `{}`）；global_impression 从"严格兜底"改为常规类别；3 个示例（global_impression / splicing_edge / 空）替换原来单一的 `{}` 示例 |
| `segment_agent/docs/segment_constitution.md`（Iter 2 补丁） | `splicing_edge` 类别加入"物体替换 / 插入"子条件：当物体的局部清晰度 / 光照方向 / 噪点 / 色调与周围环境不匹配时也算 splicing 嫌疑（针对 seed=99 漏检的"feet 被替换"案例） |

**验证结果（`validate.py --num_per_class 3`，迭代过程）**：

| 迭代 | seed | TP | FP | FN | TN | Accuracy |
|---|---|---|---|---|---|---|
| 修复前 | 42 | 0 | 0 | 1 | 3 | 0.75（含 1 错误） |
| Iter 1（重构 prompt） | 42 | 2 | 0 | 0 | 3 | **1.00** |
| Iter 1 | 7 | 2 | 0 | 0 | 3 | **1.00** |
| Iter 1 | 99 | 1 | 0 | 1 | 3 | 0.80（漏 feet-tampered） |
| Iter 2（加物体替换条件） | 99 | 1 | 0 | 1 | 3 | 0.80（待进一步） |

精度（precision）在所有迭代中保持 1.00（零误报）；召回率（recall）从修复前的 0 提升到 seed=42/7 上的 1.0、seed=99 上的 0.5。

### 2026-05-19 — Reflection prompt 改造：从"原则提炼"到"case-driven 微调"

**背景**：上一段记录的迭代优化过程中，我（手工）的诊断流程是 case-by-case 的：先看每个 bad case 是漏检还是误报、提取关键证据、引用现有 prompt 中本应触发的段落、诊断为什么没触发、然后写一个针对该盲点的最小补丁并加反向约束。这套流程比 Phase 4 reflection prompt 现写的"识别 common pattern + 提炼 1-3 条原则"具体得多。用户敏锐指出这才是 reflection agent 应该跑的流程。

**改动**：
- `reflection_agent/reflection.py:_reflection_system_prompt`：整体重写，把分析流程从"common pattern + principles"改为强制的 **4 步独立诊断（A 失败方向 / B 关键证据与对应 anomaly_type / C 现有规则盲点（必须 quote 现有 prompt 段落） / D 最小补丁含反向约束自检）**，A-D 全部做完之后再考虑跨 case 合并。明确禁止"为了凑共性而合并不同根因 case"。
- `reflection_agent/reflection.py:_build_bad_cases_text`：bad case 文本中追加 `n_screened` 与 `Failure Stage Hint`，让 LLM 一眼看出失败是发生在筛选阶段（`n_screened=0` → 应改 `segment_constitution.md`）还是下游（`n_screened>0` → 应改 `constitution.md` 或 `report.py` 规则）。

**预期效果**：reflection 不再产出空泛的"建议增强 X 类检测"，而是产出引用了现有 prompt 文本的、有反向约束自检的、可直接被 `_parse_patch_ops` 解析为 `<replace>/<insert_after>` 的窄补丁。验证门拒绝率应该降低，accuracy 提升幅度应该提高。

**下次更新本文档的触发点**：Phase 5（如启用，做"非可疑区域采样"二次验证）、reflection 长时间运行后的指标走势、或 prompt 又一次手工大改。
