def write_text_to_file(text: str, file_path: str) -> str:
    """
    将文本写入文件
    """
    with open(file_path, "w") as f:
        f.write(text)
    return file_path