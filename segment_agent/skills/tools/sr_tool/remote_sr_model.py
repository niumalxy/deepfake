from PIL import Image
import base64
import io

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

class RemoteSuperResolutionModel:
    def __init__(self, api_url: str = None, api_key: str = None, timeout: int = 30):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self._check_availability()
    
    def _check_availability(self):
        if not HTTPX_AVAILABLE:
            print("httpx not available. Please install with: pip install httpx")
            return False
        
        if self.api_url is None:
            print("Warning: Remote API URL not configured. Please provide api_url parameter.")
            return False
        
        print(f"Remote super-resolution model configured with API URL: {self.api_url}")
        return True
    
    def _image_to_base64(self, img: Image.Image) -> str:
        """将PIL.Image转换为base64字符串"""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64
    
    def _base64_to_image(self, img_base64: str) -> Image.Image:
        """将base64字符串转换为PIL.Image"""
        img_bytes = base64.b64decode(img_base64)
        buffered = io.BytesIO(img_bytes)
        img = Image.open(buffered)
        return img
    
    def super_resolution(self, img: Image.Image, scale: int = 4) -> Image.Image:
        """
        使用远程超分辨率API增强图像分辨率
        
        :param img: PIL.Image对象，输入的低分辨率图像
        :param scale: 放大倍数，默认为4
        :return: PIL.Image对象，超分辨率处理后的图像
        """
        if not self._check_availability():
            print("Remote super-resolution not available. Using high-quality interpolation as fallback.")
            width, height = img.size
            return img.resize((width * scale, height * scale), Image.LANCZOS)
        
        try:
            img_base64 = self._image_to_base64(img)
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "image": img_base64,
                "scale": scale
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                
                if "image" in result:
                    output_img = self._base64_to_image(result["image"])
                    print(f"Remote super-resolution processed successfully")
                    return output_img
                else:
                    print(f"Unexpected response format: {result}")
                    raise ValueError("Invalid response format")
        
        except httpx.HTTPError as e:
            print(f"HTTP error during remote super-resolution: {e}")
            print("Falling back to high-quality interpolation")
        except Exception as e:
            print(f"Error during remote super-resolution processing: {e}")
            print("Falling back to high-quality interpolation")
        
        width, height = img.size
        return img.resize((width * scale, height * scale), Image.LANCZOS)
    
    def super_resolution_save(self, img: Image.Image, output_path: str, scale: int = 4) -> None:
        """
        使用远程超分辨率API增强图像分辨率并保存到指定路径
        
        :param img: PIL.Image对象，输入的低分辨率图像
        :param output_path: 输出图像的保存路径
        :param scale: 放大倍数，默认为4
        """
        output_img = self.super_resolution(img, scale)
        output_img.save(output_path)
        print(f"Remote super-resolution image saved to {output_path}")
