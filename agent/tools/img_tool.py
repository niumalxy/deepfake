# 根据左上角和右下角坐标裁剪图片
def crop_img(img, top_left, bottom_right):
    """
    :param img: PIL.Image对象
    :param top_left: 左上角坐标 (x, y)
    :param bottom_right: 右下角坐标 (x, y)
    :return: 裁剪后的PIL.Image对象
    """
    return img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

# 调整图片大小
def resize_img(img, size=(256, 256)):
    """
    :param img: PIL.Image对象
    :param size: 目标大小 (width, height)
    :return: 调整后的PIL.Image对象
    """
    return img.resize(size)
