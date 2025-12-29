import base64
from PIL import Image

# 转base64
def img_to_base64(img_path: str) -> str:
    """
    将PIL.Image转换为base64编码的字符串
    """
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64