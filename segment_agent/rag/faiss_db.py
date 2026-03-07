import faiss
import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple
from logger import logs

class FaissManager:
    """FAISS 向量数据库管理类"""
    
    def __init__(self, index_dir: str = "db/faiss_data"):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "img_index.faiss")
        self.mapping_path = os.path.join(index_dir, "faiss_mapping.json")
        self.dimension = 512  # CLIP vit-base-patch32 embedding dim
        
        # Mapping from faiss internal ID to task_id
        self.id_to_task: Dict[int, str] = {}
        self.index = None
        
        self._init_db()

    def _init_db(self):
        """初始化或加载现有的FAISS索引和映射"""
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    # JSON keys are strings, convert back to int
                    str_mapping = json.load(f)
                    self.id_to_task = {int(k): v for k, v in str_mapping.items()}
                logs.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                logs.error(f"Failed to load existing FAISS index: {e}. Reinitializing.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """创建一个新的IndexFlatIP (Inner Product，因为向量已L2归一化，相当于余弦相似度)"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_task = {}
        logs.info("Created new FAISS inner-product index.")

    def _save_data(self):
        """持久化数据到磁盘"""
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_task, f, ensure_ascii=False)

    def insert_vector(self, embedding_np: np.ndarray, task_id: str) -> bool:
        """
        向数据库中插入新的向量
        
        Args:
            embedding_np (np.ndarray): 1D embedded vector
            task_id (str): MongoDB中的对应task_id
            
        Returns:
            bool: 插入是否成功
        """
        try:
            # Check length
            if len(embedding_np) != self.dimension:
                logs.error(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding_np)}")
                return False
                
            # Expand to 2D for FAISS add
            vec_2d = np.expand_dims(embedding_np, axis=0)
            
            # The next ID will be the current ntotal
            current_id = self.index.ntotal
            self.index.add(vec_2d)
            self.id_to_task[current_id] = task_id
            
            self._save_data()
            logs.info(f"Inserted vector for task_id {task_id} at faiss_id {current_id}")
            return True
        except Exception as e:
            logs.error(f"Error inserting vector to FAISS: {e}")
            return False

    def search_similar(self, embedding_np: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        搜索相似的向量
        
        Args:
            embedding_np (np.ndarray): 查询向量
            top_k (int): 返回结果数量
            
        Returns:
            List[Tuple[str, float]]: (task_id, 相似度得分) 的列表
        """
        if self.index.ntotal == 0:
            return []
            
        try:
            # Expand to 2D
            vec_2d = np.expand_dims(embedding_np, axis=0)
            
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(vec_2d, k)
            
            results = []
            # indices and distances are 2D arrays (1, k)
            for i in range(k):
                idx = int(indices[0][i])
                score = float(distances[0][i])
                if idx in self.id_to_task:
                    results.append((self.id_to_task[idx], score))
                    
            return results
        except Exception as e:
            logs.error(f"Error searching FAISS: {e}")
            return []

# 导出单例用于直接调用
faiss_manager = FaissManager()
