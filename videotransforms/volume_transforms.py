import numpy as np
from PIL import Image
import torch

from videotransforms.utils import images as imageutils

class ClipToTensor(object):
    """
    Convert a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0].

    Args:
        channel_nb (int): Number of channels in the input images. Defaults to 3.
        div_255 (bool): If True, divide pixel values by 255 to convert from range [0, 255] 
        to [0, 1.0]. Defaults to True.
        numpy (bool): If True, return the output as a numpy array. If False, return the 
        output as a torch tensor. Defaults to False.

    Methods:
        __call__(self, clip):
            Converts the input clip to a torch.FloatTensor or numpy array.

    Example:
        >>> transform = ClipToTensor(channel_nb=3, div_255=True, numpy=False)
        >>> tensor_clip = transform(clip)

    Attributes:
        channel_nb (int): Number of channels in the input images.
        div_255 (bool): Whether to divide pixel values by 255.
        numpy (bool): Whether the output should be a numpy array.
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb = channel_nb
        self.div_255 = div_255
        self.numpy = numpy

    def __call__(self, clip):
        """
        Converts the input clip to a torch.FloatTensor or numpy array.

        Args:
            clip (list of numpy.ndarray or PIL.Image): List of images to be converted. 
            Each image should be either a numpy.ndarray in the format (H, W, C) or 
            a PIL.Image.

        Returns:
            torch.FloatTensor or numpy.ndarray: Converted clip with shape 
            (C x m x H x W) and values in the range [0, 1.0] if div_255 is True, 
            otherwise in the range [0, 255] for numpy arrays.

        Raises:
            TypeError: If the input clip is not a list of numpy.ndarray or PIL.Image.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, f'Got {ch} instead of {self.channel_nb} channels'
        elif isinstance(clip[0], Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError(f'Expected numpy.ndarray or PIL.Image but got list of {type(clip[0])}')

        np_clip = np.zeros([self.channel_nb, len(clip), h, w])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError(f'Expected numpy.ndarray or PIL.Image but got list of {type(clip[0])}')
            img = imageutils.convert_img(img)
            np_clip[:, img_idx, :, :] = img

        if self.numpy:
            if self.div_255:
                np_clip = np_clip / 255
            return np_clip
        else:
            tensor_clip = torch.from_numpy(np_clip)
            if not isinstance(tensor_clip, torch.FloatTensor):
                tensor_clip = tensor_clip.float()
            if self.div_255:
                tensor_clip = tensor_clip.div(255)
            return tensor_clip


class ToTensor(object):
    """
    Converts a numpy array to a torch tensor.

    Methods:
        __call__(self, array):
            Converts the input numpy array to a torch tensor.

    Example:
        >>> transform = ToTensor()
        >>> tensor = transform(array)
    """

    def __call__(self, array):
        """
        Converts the input numpy array to a torch tensor.

        Args:
            array (numpy.ndarray): The input numpy array.

        Returns:
            torch.Tensor: The converted torch tensor.
        """
        tensor = torch.from_numpy(array)
        return tensor
