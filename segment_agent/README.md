# Segment Agent - 深度伪造图像检测Agent

## 概述

Segment Agent是一个用于检测深度伪造图像的智能Agent，它通过以下步骤分析图像：

1. **图像内容提取** (`img_content`): 识别图像中可能为伪造的区域
2. **图像分割** (`img_segment`): 根据坐标裁剪可疑区域
3. **部分图像分析** (`img_part_analysis`): 逐一对分割的部分进行深度伪造检测

## 工作流程

```
START -> init -> img_content -> img_cropping -> img_part_analysis -> END
                    |               |
                    v               v
                  (无区域)        (继续分析)
                    |               |
                    v               v
                   END         img_part_analysis (循环)
```

## 节点说明

### 1. init
- 初始化状态，存储原始图像
- 设置初始参数

### 2. img_content (extract_suspicious_regions)
- 使用LLM分析图像，识别可疑区域
- 返回可疑区域的坐标和描述
- 最多识别8个可疑区域

### 3. img_cropping (crop_image_by_coords)
- 根据坐标裁剪图像
- 保存裁剪后的图像到文件系统
- 创建CroppedImg对象

### 4. img_part_analysis (analyze_partial_image)
- 逐一对裁剪的图像进行分析
- 使用LLM进行深度伪造检测
- 支持调用图像处理工具辅助分析
- 循环直到所有图像分析完成

## 状态定义

### AgentState
```python
class AgentState(TypedDict):
    status: AgentStatus                    # 当前状态
    content_messages: List[BaseMessage]    # 内容提取消息
    analysis_messages: List[BaseMessage]   # 分析消息
    origin_img: Image.Image                 # 原始图像
    cropped_imgs: List[CroppedImg]          # 裁剪后的图像列表
    cropping_imgs: List[CroppingImg]        # 待裁剪的图像列表
    log_id: str                             # 日志ID
    current_analysis_idx: int              # 当前分析索引
```

### CroppingImg
```python
class CroppingImg(TypedDict):
    top_left: tuple[int, int]              # 左上角坐标
    bottom_right: tuple[int, int]          # 右下角坐标
    description: str                       # 描述
    save_path: str                         # 保存路径
```

### CroppedImg
```python
class CroppedImg(TypedDict):
    save_path: str                         # 保存路径
    description: str                       # 描述
    is_done: bool = False                  # 是否完成分析
    analysis_result: str = ""              # 分析结果
```

## Prompt说明

### IMAGE_ANALYSIS_PROMPT
用于识别图像中可能伪造的区域，重点关注：
- 恐怖谷效应特征
- 面部、手部等关键区域异常
- AI生成图像特有特征
- 最多识别8个区域

### PARTIAL_IMAGE_ANALYSIS_PROMPT
用于对裁剪的图像部分进行深度伪造检测，基于constitution.md中的分析准则：
- 视觉伪影与纹理质量
- 解剖学与生物学一致性
- 恐怖谷效应
- 光照、阴影与物理
- 上下文与背景
- 其他因素

## 使用方法

### 创建工作流
```python
from PIL import Image
from segment_agent.graph.workflow import create_graph

# 加载图像
img = Image.open("path/to/image.jpg")

# 创建工作流
task_id = "task_001"
graph = create_graph(task_id, img)

# 执行工作流
initial_state = {
    "status": None,
    "content_messages": [],
    "analysis_messages": [],
    "origin_img": img,
    "cropped_imgs": [],
    "cropping_imgs": [],
    "log_id": task_id,
    "current_analysis_idx": 0
}

result = graph.invoke(initial_state)
```

### 测试
```python
from segment_agent.test_segment_agent import test_segment_agent

test_segment_agent()
```

## 输出结果

工作流执行完成后，`result` 包含：
- `status`: 最终状态
- `cropping_imgs`: 可疑区域列表
- `cropped_imgs`: 裁剪后的图像列表，每个图像包含：
  - `save_path`: 保存路径
  - `description`: 描述
  - `is_done`: 是否完成分析
  - `analysis_result`: 分析结果（包含判断、置信度、关键指标、详细分析）

## 工具支持

分析过程中可以使用以下图像处理工具：
- crop_img: 裁剪图像
- resize_img: 调整图像大小
- rotate_img: 旋转图像
- flip_img: 翻转图像
- adjust_brightness: 调整亮度
- adjust_contrast: 调整对比度
- adjust_saturation: 调整饱和度
- adjust_sharpness: 调整锐度
- blur_img: 模糊图像
- sharpen_img: 锐化图像
- convert_to_grayscale: 转换为灰度图
- convert_to_rgb: 转换为RGB
- add_border: 添加边框
- invert_img: 反转颜色
- pad_img: 添加填充

## 注意事项

1. 确保测试图像存在于项目根目录
2. 确保LLM配置正确
3. 裁剪的图像默认保存在 `./segment_agent/nodes/img_content/docs` 目录
4. 日志会记录到 `log/` 目录
