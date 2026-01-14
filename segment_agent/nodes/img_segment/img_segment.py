from PIL import Image

def crop_image_by_coords(img: Image.Image, top_left: tuple, bottom_right: tuple) -> Image.Image:
    """
    根据左上角和右下角坐标切割图片
    
    Args:
        img: PIL.Image 对象
        top_left: 左上角坐标 (x1, y1)
        bottom_right: 右下角坐标 (x2, y2)
    
    Returns:
        PIL.Image: 切割后的图片
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open('example.jpg')
        >>> cropped = crop_image_by_coords(img, (100, 100), (300, 300))
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    box = (x1, y1, x2, y2)
    return img.crop(box)
