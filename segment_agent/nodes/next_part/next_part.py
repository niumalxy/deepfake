def next_part(state):
    current_idx = state.get('current_img_idx', 0)
    state["cropped_imgs"][current_idx]["analysis_result"] = state["analysis_messages"][-1].content
    state["cropped_imgs"][current_idx]["is_done"] = True
    state["current_img_idx"] = current_idx + 1
    state["analysis_messages"] = []
    return state