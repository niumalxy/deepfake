import logging
import os
import contextvars
import contextlib
from datetime import datetime

logs = logging.getLogger(__name__)
logs.setLevel(logging.INFO)

# 定义上下文变量
context_var = contextvars.ContextVar("context_var", default=None)

class ContextFilter(logging.Filter):
    def filter(self, record):
        ctx = context_var.get()
        log_id = "-"
        if ctx:
            if isinstance(ctx, dict):
                log_id = ctx.get("log_id", "-")
            else:
                log_id = getattr(ctx, "log_id", "-")
        record.log_id = log_id
        return True

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s - [%(log_id)s] - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 添加过滤器
context_filter = ContextFilter()
logs.addFilter(context_filter)
console_handler.addFilter(context_filter)

# 添加控制台处理器到日志记录器
logs.addHandler(console_handler)

# 保存路径
log_file_path = f"log/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
if not os.path.exists("log"):
    os.makedirs("log")

# 创建文件处理器
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
file_handler.addFilter(context_filter)

# 添加文件处理器到日志记录器
logs.addHandler(file_handler)


def set_context(ctx):
    """设置上下文"""
    return context_var.set(ctx)

def get_context():
    """获取上下文"""
    return context_var.get()

@contextlib.contextmanager
def scoped_context(ctx):
    """上下文管理器"""
    token = set_context(ctx)
    try:
        yield
    finally:
        context_var.reset(token)


