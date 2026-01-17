from PIL import Image, ImageEnhance, ImageFilter

def crop_img(img, top_left, bottom_right):
    """
    Crop the image based on top-left and bottom-right coordinates.
    :param img: PIL.Image object
    :param top_left: Top-left coordinates (x, y)
    :param bottom_right: Bottom-right coordinates (x, y)
    :return: Cropped PIL.Image object
    """
    width, height = img.size
    
    # Check for out of bounds
    if (top_left[0] < 0 or top_left[1] < 0 or 
        bottom_right[0] > width or bottom_right[1] > height):
        raise ValueError(
            f"Crop coordinates out of bounds. "
            f"Image size is {width}x{height}, but requested crop is "
            f"from {top_left} to {bottom_right}."
        )

    return img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

def resize_img(img, size=(256, 256)):
    """
    Resize the image to the specified size.
    :param img: PIL.Image object
    :param size: Target size (width, height)
    :return: Resized PIL.Image object
    """
    return img.resize(size)

def rotate_img(img, angle=90, expand=False):
    """
    Rotate the image by the specified angle.
    :param img: PIL.Image object
    :param angle: Rotation angle in degrees (counter-clockwise). Default is 90.
    :param expand: If True, expands the output image to fit the rotated image. Default is False.
    :return: Rotated PIL.Image object
    """
    return img.rotate(angle, expand=expand)

def flip_img(img, direction="horizontal"):
    """
    Flip the image horizontally or vertically.
    :param img: PIL.Image object
    :param direction: Flip direction - "horizontal" or "vertical". Default is "horizontal".
    :return: Flipped PIL.Image object
    """
    if direction == "horizontal":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == "vertical":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

def adjust_brightness(img, factor=1.0):
    """
    Adjust the brightness of the image.
    :param img: PIL.Image object
    :param factor: Brightness factor. 1.0 gives original image, 0.0 gives black image, 2.0 gives twice brightness. Default is 1.0.
    :return: Brightness-adjusted PIL.Image object
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_contrast(img, factor=1.0):
    """
    Adjust the contrast of the image.
    :param img: PIL.Image object
    :param factor: Contrast factor. 1.0 gives original image, 0.0 gives solid gray image, 2.0 gives twice contrast. Default is 1.0.
    :return: Contrast-adjusted PIL.Image object
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def adjust_saturation(img, factor=1.0):
    """
    Adjust the saturation of the image.
    :param img: PIL.Image object
    :param factor: Saturation factor. 1.0 gives original image, 0.0 gives grayscale, 2.0 gives twice saturation. Default is 1.0.
    :return: Saturation-adjusted PIL.Image object
    """
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def adjust_sharpness(img, factor=1.0):
    """
    Adjust the sharpness of the image.
    :param img: PIL.Image object
    :param factor: Sharpness factor. 1.0 gives original image, 0.0 gives blurred image, 2.0 gives sharpened image. Default is 1.0.
    :return: Sharpness-adjusted PIL.Image object
    """
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def blur_img(img, radius=2):
    """
    Apply Gaussian blur to the image.
    :param img: PIL.Image object
    :param radius: Blur radius. Larger radius gives more blur. Default is 2.
    :return: Blurred PIL.Image object
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def sharpen_img(img):
    """
    Apply sharpening filter to the image.
    :param img: PIL.Image object
    :return: Sharpened PIL.Image object
    """
    return img.filter(ImageFilter.SHARPEN)

def convert_to_grayscale(img):
    """
    Convert the image to grayscale.
    :param img: PIL.Image object
    :return: Grayscale PIL.Image object
    """
    return img.convert("L")

def convert_to_rgb(img):
    """
    Convert the image to RGB mode.
    :param img: PIL.Image object
    :return: RGB PIL.Image object
    """
    return img.convert("RGB")

def add_border(img, border_width=10, border_color="black"):
    """
    Add a border around the image.
    :param img: PIL.Image object
    :param border_width: Width of the border in pixels. Default is 10.
    :param border_color: Color of the border. Can be a color name, hex code, or RGB tuple. Default is "black".
    :return: PIL.Image object with border
    """
    from PIL import ImageOps
    return ImageOps.expand(img, border=border_width, fill=border_color)

def invert_img(img):
    """
    Invert the colors of the image.
    :param img: PIL.Image object
    :return: Inverted PIL.Image object
    """
    from PIL import ImageOps
    return ImageOps.invert(img)

def pad_img(img, padding=10, color="white"):
    """
    Add padding around the image.
    :param img: PIL.Image object
    :param padding: Padding size in pixels. Can be a single value or tuple (left, top, right, bottom). Default is 10.
    :param color: Padding color. Can be a color name, hex code, or RGB tuple. Default is "white".
    :return: Padded PIL.Image object
    """
    from PIL import ImageOps
    return ImageOps.expand(img, border=padding, fill=color)
