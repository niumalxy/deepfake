import time
import random

def generate_id() -> str:
    """
    生成一个唯一的ID, 规则为：时间戳+4位随机数
    """
    timestamp = int(time.time())
    random_num = random.randint(0, 9999)
    return f"{timestamp}{random_num:04d}"
