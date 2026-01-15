import base64
from PIL import Image
from io import BytesIO
from typing import Optional

def img_path_to_base64(img_path: str) -> str:
    """
    将图片文件路径转换为base64编码的字符串
    """
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def img_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """
    将PIL.Image转换为base64编码的字符串
    
    Args:
        img: PIL.Image对象
        format: 图像格式，默认为JPEG
    
    Returns:
        base64编码的字符串
    """
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def base64_to_img(img_base64: str) -> Optional[Image.Image]:
    """
    将base64编码的字符串转换为PIL.Image
    
    Args:
        img_base64: base64编码的字符串
    
    Returns:
        PIL.Image对象
    """
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_bytes))
    return img