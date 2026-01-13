from PIL import Image
from segment_agent.skills.tools.sr_tool.local_sr_model import LocalSuperResolutionModel
from segment_agent.skills.tools.sr_tool.remote_sr_model import RemoteSuperResolutionModel
from segment_agent.skills.tools.sr_tool import sr_model

def get_local_sr_model(model_path: str = None, model_type: str = "EDSR") -> LocalSuperResolutionModel:
    global local_sr_model
    if local_sr_model is None:
        local_sr_model = LocalSuperResolutionModel(model_path, model_type)
    return local_sr_model

def get_remote_sr_model(api_url: str = None, api_key: str = None, timeout: int = 30) -> RemoteSuperResolutionModel:
    global remote_sr_model
    if remote_sr_model is None:
        remote_sr_model = RemoteSuperResolutionModel(api_url, api_key, timeout)
    return remote_sr_model

def super_resolution(img: Image.Image, scale: int = 4) -> Image.Image:
    """
    使用超分辨率模型增强图像分辨率
    :param img: PIL.Image对象，输入的低分辨率图像
    :param scale: 放大倍数，默认为4
    :return: PIL.Image对象，超分辨率处理后的图像
    """
    return sr_model.super_resolution(img, scale)

def super_resolution_save(img: Image.Image, output_path: str, scale: int = 4, mode: str = "local", **kwargs) -> None:
    """
    使用超分辨率模型增强图像分辨率并保存到指定路径（支持本地和远程模式）
    
    :param img: PIL.Image对象，输入的低分辨率图像
    :param output_path: 输出图像的保存路径
    :param scale: 放大倍数，默认为4
    :param mode: 调用模式，"local" 或 "remote"，默认为 "local"
    :param kwargs: 其他参数
        - model_path: 本地模型文件路径（mode="local"时使用）
        - model_type: 本地模型类型（mode="local"时使用）
        - api_url: 远程API URL（mode="remote"时使用）
        - api_key: 远程API密钥（mode="remote"时使用）
        - timeout: 远程请求超时时间（mode="remote"时使用）
    """
    if mode == "local":
        model = get_local_sr_model(
            model_path=kwargs.get("model_path"),
            model_type=kwargs.get("model_type", "EDSR")
        )
        model.super_resolution_save(img, output_path, scale)
    elif mode == "remote":
        model = get_remote_sr_model(
            api_url=kwargs.get("api_url"),
            api_key=kwargs.get("api_key"),
            timeout=kwargs.get("timeout", 30)
        )
        model.super_resolution_save(img, output_path, scale)
    else:
        print(f"Unknown mode: {mode}. Using local mode as fallback.")
        model = get_local_sr_model()
        model.super_resolution_save(img, output_path, scale)
