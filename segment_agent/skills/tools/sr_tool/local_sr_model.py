from PIL import Image
import numpy as np
from typing import Optional
import os

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class LocalSuperResolutionModel:
    def __init__(self, model_path: str = None, model_type: str = "EDSR"):
        self.model = None
        self.model_path = model_path
        self.model_type = model_type
        self._load_model()
    
    def _load_model(self):
        if not CV2_AVAILABLE:
            print("OpenCV not available. Using bicubic interpolation.")
            return
        
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            
            if self.model_path is None:
                weights_dir = os.path.join(os.path.dirname(__file__), "weights")
                os.makedirs(weights_dir, exist_ok=True)
                
                model_files = {
                    "EDSR": "EDSR_x4.pb",
                    "ESPCN": "ESPCN_x4.pb",
                    "FSRCNN": "FSRCNN_x4.pb",
                    "LAPSRN": "LAPSRN_x4.pb"
                }
                
                if self.model_type in model_files:
                    self.model_path = os.path.join(weights_dir, model_files[self.model_type])
            
            if self.model_path and os.path.exists(self.model_path):
                sr.readModel(self.model_path)
                sr.setModel(self.model_type.lower(), 4)
                self.model = sr
                print(f"Local super-resolution model ({self.model_type}) loaded successfully from {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}. Using bicubic interpolation as fallback.")
                self.model = None
        except AttributeError:
            print("OpenCV dnn_superres not available in this version. Using bicubic interpolation as fallback.")
            self.model = None
        except Exception as e:
            print(f"Error loading local super-resolution model: {e}")
            print("Using bicubic interpolation as fallback")
            self.model = None
    
    def super_resolution(self, img: Image.Image, scale: int = 4) -> Image.Image:
        """
        使用本地超分辨率模型增强图像分辨率
        
        :param img: PIL.Image对象，输入的低分辨率图像
        :param scale: 放大倍数，默认为4
        :return: PIL.Image对象，超分辨率处理后的图像
        """
        if self.model is None:
            print("Local super-resolution model not available. Using high-quality interpolation as fallback.")
            width, height = img.size
            return img.resize((width * scale, height * scale), Image.LANCZOS)
        
        try:
            img_np = np.array(img)
            
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 4:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            result = self.model.upsample(img_np)
            
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            output_img = Image.fromarray(result)
            
            return output_img
        except Exception as e:
            print(f"Error during local super-resolution processing: {e}")
            print("Falling back to high-quality interpolation")
            width, height = img.size
            return img.resize((width * scale, height * scale), Image.LANCZOS)
    
    def super_resolution_save(self, img: Image.Image, output_path: str, scale: int = 4) -> None:
        """
        使用本地超分辨率模型增强图像分辨率并保存到指定路径
        
        :param img: PIL.Image对象，输入的低分辨率图像
        :param output_path: 输出图像的保存路径
        :param scale: 放大倍数，默认为4
        """
        output_img = self.super_resolution(img, scale)
        output_img.save(output_path)
        print(f"Local super-resolution image saved to {output_path}")
