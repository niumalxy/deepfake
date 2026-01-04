from starlette.middleware.base import BaseHTTPMiddleware
from logger import task_id_var
from utils.idgen import generate_id

def init(app):
    class TaskIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # 先判断请求是否包含task_id
            if "task_id" in request.query_params:
                task_id = request.query_params["task_id"]
            else:
                task_id = generate_id()
            token = task_id_var.set(task_id)
            try:
                response = await call_next(request)
                return response
            finally:
                task_id_var.reset(token)
    app.add_middleware(TaskIdMiddleware)