import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from logger import logs
import numpy as np

class ImgEmbedding:
    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImgEmbedding, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """初始化CLIP模型和预处理器"""
        try:
            logs.info("Initializing CLIP model for image embedding...")
            # Detect device (use CUDA if available, otherwise MPS or CPU)
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            logs.info(f"Using device: {self.device}")
            
            # 使用 openai/clip-vit-base-patch32
            model_id = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(model_id).to(self.device)
            self._processor = CLIPProcessor.from_pretrained(model_id)
            logs.info("CLIP model loaded successfully.")
        except Exception as e:
            logs.error(f"Failed to load CLIP model: {e}")
            raise e

    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        提取图像的嵌入向量并归一化
        
        Args:
            image (Image.Image): PIL格式的图像
            
        Returns:
            np.ndarray: 归一化后的一维图像特征向量 (float32)
        """
        try:
            # 确保图像转为RGB格式
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            inputs = self._processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # CLIPModel.get_image_features needs pixel_values explicitly for image only inputs
                features_out = self._model.get_image_features(pixel_values=inputs["pixel_values"])
                
                # Handle different return types based on transformers version
                if isinstance(features_out, torch.Tensor):
                    image_features = features_out
                elif hasattr(features_out, 'image_embeds'):
                    image_features = features_out.image_embeds
                elif hasattr(features_out, 'pooler_output'):
                    # Fallback to pooler_output if it returned BaseModelOutputWithPooling
                    image_features = features_out.pooler_output
                else:
                    # Generic fallback
                    image_features = features_out[0] if isinstance(features_out, (tuple, list)) else features_out
                
            # 归一化向量 (L2 norm)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # 转储到 numpy 数组并降为 1D
            embedding_np = image_features.cpu().numpy().flatten().astype(np.float32)
            
            return embedding_np
            
        except Exception as e:
            logs.error(f"Error generating image embedding: {e}")
            raise e

# 导出单例用于直接调用
img_embedder = ImgEmbedding()
