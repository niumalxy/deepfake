def next_part(state):
    state["cropped_imgs"][state["current_analysis_idx"]]["analysis_result"] = state["analysis_messages"][-1].content
    state["cropped_imgs"][state["current_analysis_idx"]]["is_done"] = True
    state["current_analysis_idx"] += 1
    state["analysis_messages"] = []
    return state