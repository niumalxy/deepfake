def get_partial_image_analysis_prompt() -> str:
    with open("segment_agent/docs/constitution.md", "r") as f:
        PARTIAL_IMAGE_ANALYSIS_PROMPT = f.read()
    return PARTIAL_IMAGE_ANALYSIS_PROMPT
