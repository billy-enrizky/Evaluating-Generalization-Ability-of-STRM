def normalize(tensor, mean, std):
    """
    Normalizes a tensor with the given mean and standard deviation.

    This function normalizes the input tensor by subtracting the mean and dividing by the standard deviation.
    This is typically used for preprocessing input data before feeding it into a neural network.

    Args:
        tensor (Tensor): The input tensor to normalize. This tensor can be of any shape.
        mean (float or Tensor): The mean to subtract from the tensor. If a float, the same mean is subtracted 
                                from all elements. If a tensor, it should have the same number of elements as the 
                                input tensor and the mean is subtracted element-wise.
        std (float or Tensor): The standard deviation to divide the tensor by. If a float, the same standard 
                               deviation is used for all elements. If a tensor, it should have the same number of 
                               elements as the input tensor and the division is element-wise.

    Returns:
        Tensor: The normalized tensor, where each element is adjusted to have the specified mean and standard deviation.

    Example:
        >>> import torch
        >>> tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> mean = 2.5
        >>> std = 1.0
        >>> normalize(tensor, mean, std)
        tensor([-1.5000, -0.5000,  0.5000,  1.5000])

    Note:
        The operation is performed in-place on the input tensor. If you want to preserve the original tensor,
        you should clone it before passing to this function.

    """
    tensor.sub_(mean).div_(std)
    return tensor
