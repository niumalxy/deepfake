def next_part(state):
    current_idx = state.get('current_img_idx', 0)
    cropped_imgs = state.get("cropped_imgs", [])
    
    # 确保我们有analysis_messages
    if state.get("analysis_messages"):
        cropped_imgs[current_idx]["analysis_result"] = state["analysis_messages"][-1].content
    
    cropped_imgs[current_idx]["is_done"] = True
    
    return {
        "cropped_imgs": cropped_imgs,
        "current_img_idx": current_idx + 1,
        "analysis_messages": []  # 清空消息历史
    }