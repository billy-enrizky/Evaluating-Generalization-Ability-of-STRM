import random
import torch
from videotransforms.utils import functional as F

class Normalize(object):
    """
    Normalize a tensor image with mean and standard deviation.

    This transformation normalizes each channel of the input tensor using the provided mean and standard deviation.
    The formula used for normalization is: channel = (channel - mean) / std

    Args:
        mean (float or list of float): Mean value(s) for normalization. If a single float is provided, the same mean 
                                       is applied to all channels. If a list of floats is provided, it should have the 
                                       same length as the number of channels in the tensor.
        std (float or list of float): Standard deviation value(s) for normalization. If a single float is provided, 
                                      the same std is applied to all channels. If a list of floats is provided, it should 
                                      have the same length as the number of channels in the tensor.

    Methods:
        __call__(tensor): Normalizes the input tensor.

    Example:
        >>> import torch
        >>> tensor = torch.rand(3, 100, 100) * 255  # Random tensor with 3 channels
        >>> transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        >>> normalized_tensor = transform(tensor)
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Normalizes the input tensor.

        Args:
            tensor (Tensor): Tensor of stacked images or image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized tensor.

        Raises:
            TypeError: If the input is not a torch.Tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Expected input type torch.Tensor but got {0}'.format(type(tensor)))

        return F.normalize(tensor, self.mean, self.std)


class SpatialRandomCrop(object):
    """
    Crops a random spatial crop in a spatio-temporal input.

    This transformation crops a random region from a 4D tensor of shape [Channel, Time, Height, Width].
    The size of the cropped region is specified by the user.

    Args:
        size (tuple): The desired size of the crop in the format (height, width).

    Methods:
        __call__(tensor): Applies the random spatial crop to the input tensor.

    Example:
        >>> import torch
        >>> tensor = torch.rand(3, 10, 100, 100)  # Random tensor with 3 channels, 10 time steps, 100x100 spatial dimensions
        >>> transform = SpatialRandomCrop(size=(50, 50))
        >>> cropped_tensor = transform(tensor)
        >>> cropped_tensor.shape
        torch.Size([3, 10, 50, 50])
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        """
        Applies the random spatial crop to the input tensor.

        Args:
            tensor (Tensor or numpy.ndarray): The input tensor of shape [Channel, Time, Height, Width].

        Returns:
            Tensor or numpy.ndarray: The cropped tensor.

        Raises:
            ValueError: If the crop size is larger than the input tensor size.
            TypeError: If the input is not a torch.Tensor or numpy.ndarray.
        """
        if not (isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray)):
            raise TypeError('Expected input type torch.Tensor or numpy.ndarray but got {0}'.format(type(tensor)))

        h, w = self.size
        _, _, tensor_h, tensor_w = tensor.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger than cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, tensor_w - w)
        y1 = random.randint(0, tensor_h - h)
        cropped = tensor[:, :, y1:y1 + h, x1:x1 + w]
        return cropped
