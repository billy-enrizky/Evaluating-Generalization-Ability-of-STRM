import numpy as np

def convert_img(img):
    """
    Converts an image from (H, W, C) format to (C, W, H) format.

    This function transposes a given image array from height-width-channel (H, W, C) format 
    to channel-width-height (C, W, H) format, which is often required for input into deep learning 
    frameworks such as PyTorch. If the input image is already in (H, W) format (grayscale image), 
    it adds an extra dimension to convert it to (C, W, H) format.

    Args:
        img (numpy.ndarray): The input image array. This can be a 3-dimensional array (H, W, C) for 
                             color images or a 2-dimensional array (H, W) for grayscale images.

    Returns:
        numpy.ndarray: The transposed image array in (C, W, H) format. For grayscale images, it 
                       returns a 3-dimensional array with a single channel.

    Example:
        >>> import numpy as np
        >>> img = np.random.rand(100, 200, 3)  # Random color image
        >>> converted_img = convert_img(img)
        >>> converted_img.shape
        (3, 200, 100)

        >>> img_gray = np.random.rand(100, 200)  # Random grayscale image
        >>> converted_img_gray = convert_img(img_gray)
        >>> converted_img_gray.shape
        (1, 200, 100)

    Note:
        This function assumes that the input image is either a 2D grayscale image or a 3D color image.
        It does not perform any validation on the input shape beyond checking its length.
    """
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    elif len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img

