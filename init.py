from starlette.middleware.base import BaseHTTPMiddleware
from logger import context_var
from utils.idgen import generate_id

def init(app):
    class TaskIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # 先判断请求是否包含log_id
            if "log_id" in request.query_params:
                log_id = request.query_params["log_id"]
            else:
                log_id = generate_id()
            token = context_var.set({"log_id": log_id})
            try:
                response = await call_next(request)
                return response
            finally:
                context_var.reset(token)
    app.add_middleware(TaskIdMiddleware)