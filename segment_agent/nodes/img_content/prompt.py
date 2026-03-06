def get_image_analysis_prompt() -> str:
    with open("segment_agent/docs/segment_constitution.md", "r") as f:
        IMAGE_ANALYSIS_PROMPT = f.read()
    return IMAGE_ANALYSIS_PROMPT