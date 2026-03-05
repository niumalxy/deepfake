with open("segment_agent/docs/segment_constitution.md", "r") as f:
    PARTIAL_IMAGE_ANALYSIS_PROMPT = f.read()

def get_partial_image_analysis_prompt() -> str:
    return PARTIAL_IMAGE_ANALYSIS_PROMPT
