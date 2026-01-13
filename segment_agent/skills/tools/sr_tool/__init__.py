# 超分辨率模型加载
from nt import environ
from segment_agent.skills.tools.sr_tool.sr_tool import (
    get_local_sr_model,
    get_remote_sr_model,
    super_resolution,
    super_resolution_save
)
from segment_agent.skills.tools.sr_tool.local_sr_model import LocalSuperResolutionModel
from segment_agent.skills.tools.sr_tool.remote_sr_model import RemoteSuperResolutionModel

sr_model = None

# 根据环境变量 SR_MODEL 选择本地或远程模型
# 可选值: "local" 或 "remote"，默认为 "local"
sr_mode = environ.get("SR_MODEL", "local")

if sr_mode == "local":
    sr_model = get_local_sr_model()
elif sr_mode == "remote":
    sr_model = get_remote_sr_model()
else:
    print(f"Unknown SR_MODEL value: {sr_mode}. Using local mode as default.")
    sr_model = get_local_sr_model()
