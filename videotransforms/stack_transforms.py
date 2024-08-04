import numpy as np
import PIL
import torch
from videotransforms.utils import images as imageutils

class ToStackedTensor(object):
    """
    Converts a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    or PIL Images to a torch.FloatTensor of shape (m*C x H x W)
    in the range [0, 1.0].

    This transformation is useful for preparing a list of images (e.g., video frames) 
    for input into a neural network that expects a stacked tensor format.

    Attributes:
        channel_nb (int): The number of channels in the input images. Default is 3 (for RGB images).

    Methods:
        __call__(clip): Converts the input clip (list of images) to a stacked tensor.
    """

    def __init__(self, channel_nb=3):
        """
        Initializes the ToStackedTensor object with the number of channels.

        Args:
            channel_nb (int): The number of channels in the input images. Default is 3.
        """
        self.channel_nb = channel_nb

    def __call__(self, clip):
        """
        Converts the input clip (list of images) to a stacked tensor.

        Args:
            clip (list of numpy.ndarray or PIL.Image.Image): The clip (list of images) to be converted to a tensor. 
                                                             Each image should be in the format H x W x C.

        Returns:
            torch.FloatTensor: A tensor of shape (m*C x H x W) in the range [0, 1.0].

        Raises:
            TypeError: If the elements of the input clip are neither numpy arrays nor PIL Images.
            AssertionError: If the number of channels in the input images does not match `self.channel_nb`.

        Example:
            >>> import numpy as np
            >>> from PIL import Image
            >>> img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
            >>> clip = [img, img]
            >>> transform = ToStackedTensor(channel_nb=3)
            >>> tensor_clip = transform(clip)
            >>> tensor_clip.shape
            torch.Size([6, 100, 200])
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'got {} channels instead of {}'.format(ch, self.channel_nb)
        elif isinstance(clip[0], PIL.Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([self.channel_nb * len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, PIL.Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
            img = imageutils.convert_img(img)
            np_clip[img_idx * self.channel_nb:(img_idx + 1) * self.channel_nb, :, :] = img

        tensor_clip = torch.from_numpy(np_clip)
        return tensor_clip.float().div(255)
