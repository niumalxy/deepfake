import base64
from PIL import Image
from typing import Optional

# 转base64
def img_path_to_base64(img_path: str) -> str:
    """
    将PIL.Image转换为base64编码的字符串
    """
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def img_to_base64(img: Image.Image) -> str:
    """
    将PIL.Image转换为base64编码的字符串
    """
    img_bytes = img.tobytes(encoder_name="raw")
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def base64_to_img(img_base64: str, img_size: tuple[int, int] = (256, 256), save_path: str = None) -> Optional[Image.Image]:
    """
    将base64编码的字符串转换为PIL.Image
    """
    img_bytes = base64.b64decode(img_base64)
    img = Image.frombytes("RGB", img_size, img_bytes)
    if save_path:
        img.save(save_path)
        return
    return img