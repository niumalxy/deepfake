from typing import Dict, Any, List
from langchain_core.tools import tool
from segment_agent.rag.faiss_db import faiss_manager
from utils.img_embedding import img_embedder
from db.mongodb import get_analysis_by_task_id
from logger import logs

# We need a way to pass the origin_img to this tool since it comes from State.
# Langchain @tool doesn't naturally pass State. 
# We'll use a global variable or class property approach temporarily for the agent run, 
# or since this tool is called by the agent specifically for the CURRENT image,
# we can fetch the image from a shared context if needed.
# To keep it simple and thread-safe without full state passing, 
# we'll create a class-based tool or use a closure in the node.

def get_rag_tool(origin_img_placeholder) -> Any:
    """
    Returns a configured search_similar_images tool that has access to the current image.
    """
    
    @tool
    def search_similar_images(k: int = 3) -> str:
        """
        Search for historical deepfake analysis cases that are visually similar to the current image.
        
        Args:
            k (int): The number of similar historical cases to retrieve. (recommended: 3)
            
        Returns:
            str: A formatted string containing the descriptions, labels, and original reports of the retrieved similar cases.
                 If no cases are found, returns a message indicating so.
        """
        logs.info(f"Agent requested {k} similar images from FAISS...")
        
        try:
            # 1. Get embedding of current image
            emb = img_embedder.get_embedding(origin_img_placeholder)
            
            # 2. Search in FAISS
            results = faiss_manager.search_similar(emb, top_k=k)
            
            if not results:
                return "No similar historical cases found in the database."
                
            # 3. Retrieve detailed reports from MongoDB
            formatted_results = []
            for idx, (task_id, score) in enumerate(results):
                analysis_data = get_analysis_by_task_id(task_id)
                if not analysis_data:
                    continue
                    
                label_str = "fake" if analysis_data.get("label") == 0 else "normal"
                desc = analysis_data.get("description", "No description")
                report = analysis_data.get("report", "No report available")
                
                res_str = f"--- Case {idx + 1} (Similarity Score: {score:.4f}) ---\n"
                res_str += f"Ground Truth Label: {label_str}\n"
                res_str += f"Image Description: {desc}\n"
                res_str += f"Historical Analysis Report:\n{report}\n"
                
                formatted_results.append(res_str)
                
            if not formatted_results:
                return "Similar cases were found in FAISS, but their detailed records are missing in MongoDB."
                
            return "\n".join(formatted_results)
            
        except Exception as e:
            err_msg = f"Error during similarity search: {e}"
            logs.error(err_msg)
            return err_msg
            
    return search_similar_images
