import numbers
import numpy as np
import PIL
import torchvision

def crop_clip(clip, min_h, min_w, h, w):
    """
    Crops each image in a clip to a specified rectangle.

    Args:
        clip (list of numpy.ndarray or list of PIL.Image.Image): The list of images to be cropped. Each image can be either 
                                                                a numpy array or a PIL Image.
        min_h (int): The minimum height coordinate of the crop box.
        min_w (int): The minimum width coordinate of the crop box.
        h (int): The height of the crop box.
        w (int): The width of the crop box.

    Returns:
        list: The list of cropped images, with the same type as the input images.

    Raises:
        TypeError: If the elements of the input clip are neither numpy arrays nor PIL Images.

    Example:
        >>> import numpy as np
        >>> img = np.random.rand(100, 200, 3)
        >>> clip = [img, img]
        >>> cropped_clip = crop_clip(clip, 10, 20, 50, 60)
        >>> cropped_clip[0].shape
        (50, 60, 3)
    """
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
    return cropped

def resize_clip(clip, size, interpolation='bilinear'):
    """
    Resizes each image in a clip to the specified size.

    Args:
        clip (list of numpy.ndarray or list of PIL.Image.Image): The list of images to be resized. Each image can be either 
                                                                a numpy array or a PIL Image.
        size (int or tuple): The target size. If an integer, the smaller edge of the image will be matched to this number 
                             while maintaining the aspect ratio. If a tuple, it specifies the exact (width, height).
        interpolation (str): The interpolation method to use for resizing. Options are 'bilinear' or 'nearest'. 
                             Default is 'bilinear'.

    Returns:
        list: The list of resized images, with the same type as the input images.

    Raises:
        NotImplementedError: If resizing for numpy arrays is attempted (not implemented in this function).
        TypeError: If the elements of the input clip are neither numpy arrays nor PIL Images.

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 200))
        >>> clip = [img, img]
        >>> resized_clip = resize_clip(clip, (50, 100))
        >>> resized_clip[0].size
        (50, 100)
    """
    if isinstance(clip[0], np.ndarray):
        raise NotImplementedError
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                return clip
            new_h, new_w = get_resize_sizes(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.BILINEAR
        else:
            pil_inter = PIL.Image.NEAREST
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
    return scaled

def get_resize_sizes(im_h, im_w, size):
    """
    Calculates the new dimensions for resizing while maintaining aspect ratio.

    Args:
        im_h (int): The original height of the image.
        im_w (int): The original width of the image.
        size (int): The target size for the smaller edge of the image.

    Returns:
        tuple: A tuple (new_height, new_width) with the new dimensions of the image.

    Example:
        >>> get_resize_sizes(100, 200, 50)
        (50, 100)
    """
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow
