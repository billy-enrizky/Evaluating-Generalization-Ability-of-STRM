import numbers
import random

#import cv2
from matplotlib import pyplot as plt
import numpy as np
import PIL
import scipy
import torch
import torchvision
from PIL import Image
from . import functional as F


class Compose(object):
    """
    Composes several transforms together. This class applies a series of 
    transformations to a video clip in sequence.

    Args:
        transforms (list of ``Transform`` objects): List of transformations to 
        compose. Each transformation in the list should be callable and accept 
        a video clip as input.

    Example:
        >>> transform = Compose([
        >>>     Resize((256, 256)),
        >>>     ToTensor(),
        >>>     Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        >>> ])
        >>> transformed_clip = transform(clip)

    Methods:
        __init__(self, transforms):
            Initializes the Compose object with a list of transforms.

        __call__(self, clip):
            Applies the list of transforms to the input video clip.

    Attributes:
        transforms (list): List of transformations to apply to the video clip.

    """

    def __init__(self, transforms):
        """
        Initializes the Compose object.

        Args:
            transforms (list of ``Transform`` objects): List of transformations 
            to compose. Each transformation should be callable and accept a 
            video clip as input.
        """
        self.transforms = transforms

    def __call__(self, clip):
        """
        Applies the list of transforms to the input video clip.

        Args:
            clip (ndarray or PIL.Image): The input video clip to transform. 
            This could be a single frame (image) or a sequence of frames 
            (video).

        Returns:
            The transformed video clip, after applying all the transformations 
            in sequence.
        """
        for t in self.transforms:
            clip = t(clip)
        return clip



class RandomHorizontalFlip(object):
    """
    Horizontally flips the list of given images randomly with a probability of 0.5.

    This transform can be applied to both numpy.ndarray and PIL.Image objects. It 
    iterates through a list of images (frames of a video clip) and applies a 
    horizontal flip to each image with a 50% chance.

    Methods:
        __call__(self, clip):
            Flips the list of images horizontally with a probability of 0.5.

    Example:
        >>> transform = RandomHorizontalFlip()
        >>> flipped_clip = transform(clip)

    Attributes:
        None
    """

    def __call__(self, clip):
        """
        Horizontally flips the list of given images randomly with a probability of 0.5.

        Args:
            clip (list of PIL.Image or numpy.ndarray): List of images to be flipped.
            Each image should be either a PIL.Image or a numpy.ndarray in the format 
            (h, w, c).

        Returns:
            list of PIL.Image or numpy.ndarray: Randomly flipped clip. The type of 
            the returned clip will match the type of the input clip.

        Raises:
            TypeError: If the input clip is not a list of PIL.Image or numpy.ndarray.
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip



class RandomResize(object):
    """
    Resizes a list of (H x W x C) numpy.ndarray or PIL.Image to a final size.

    The larger the original image is, the longer it takes to interpolate. The resize 
    operation scales the images by a random factor chosen uniformly from a specified 
    range of ratios.

    Args:
        ratio (tuple of float): Range of scaling factors to choose from. The final 
        size is computed as original size multiplied by a random scaling factor chosen 
        from this range. Defaults to (3/4, 4/3).
        interpolation (str): Interpolation method to use for resizing. Can be one of 
        'nearest' or 'bilinear'. Defaults to 'nearest'.

    Methods:
        __call__(self, clip):
            Resizes the list of images using a randomly chosen scaling factor.

    Example:
        >>> transform = RandomResize(ratio=(0.8, 1.2), interpolation='bilinear')
        >>> resized_clip = transform(clip)

    Attributes:
        ratio (tuple of float): Range of scaling factors.
        interpolation (str): Interpolation method.
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        """
        Resizes the list of images using a randomly chosen scaling factor.

        Args:
            clip (list of PIL.Image or numpy.ndarray): List of images to be resized. 
            Each image should be either a PIL.Image or a numpy.ndarray in the format 
            (H, W, C).

        Returns:
            list of PIL.Image or numpy.ndarray: Resized clip. The type of the returned 
            clip will match the type of the input clip.

        Raises:
            TypeError: If the input clip is not a list of PIL.Image or numpy.ndarray.
        """
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized



class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = F.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop(object):
    """
    Extract random crop at the same location for a list of images.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
                                If an integer is provided, a square crop of size (size, size) is made.

    Attributes:
        size (tuple): Size of the crop in format (h, w).

    Methods:
        __call__(clip):
            Extracts a random crop from the input clip.

    Example:
        >>> transform = RandomCrop(size=(224, 224))
        >>> cropped_clip = transform(clip)

    Raises:
        TypeError: If the input clip is not a list of numpy.ndarray or PIL.Image.
        ValueError: If the crop size is larger than the original image size.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Extracts a random crop from the input clip.

        Args:
            clip (list of numpy.ndarray or PIL.Image): List of images to be cropped.
                                                       Each image should be either a numpy.ndarray
                                                       in the format (h, w, c) or a PIL.Image.

        Returns:
            list of numpy.ndarray or PIL.Image: Cropped list of images.

        Raises:
            TypeError: If the input clip is not a list of numpy.ndarray or PIL.Image.
            ValueError: If the crop size is larger than the original image size.
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError(f'Expected numpy.ndarray or PIL.Image but got list of {type(clip[0])}')

        if w > im_w or h > im_h:
            error_msg = (
                f'Initial image size should be larger than cropped size but got cropped sizes : '
                f'({w}, {h}) while initial image is ({im_w}, {im_h})'
            )
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class RandomRotation(object):
    """
    Rotate entire clip randomly by a random angle within given bounds.

    Args:
        degrees (sequence or int): Range of degrees to select from.
                                   If degrees is a number instead of a sequence like (min, max),
                                   the range of degrees will be (-degrees, +degrees).

    Attributes:
        degrees (tuple): Range of degrees to select from.

    Methods:
        __call__(clip):
            Rotates each image in the clip by a random angle within the specified bounds.

    Example:
        >>> transform = RandomRotation(degrees=(30, 60))
        >>> rotated_clip = transform(clip)

    Raises:
        ValueError: If degrees is a single number and is negative.
                    If degrees is a sequence and is not of length 2.
        TypeError: If the input clip is not a list of numpy.ndarray or PIL.Image.
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number, it must be positive.')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence, it must be of length 2.')
        self.degrees = degrees

    def __call__(self, clip):
        """
        Rotates each image in the clip by a random angle within the specified bounds.

        Args:
            clip (list of numpy.ndarray or PIL.Image): List of images to be rotated.
                                                       Each image should be either a numpy.ndarray
                                                       in the format (h, w, c) or a PIL.Image.

        Returns:
            list of numpy.ndarray or PIL.Image: Rotated list of images.

        Raises:
            TypeError: If the input clip is not a list of numpy.ndarray or PIL.Image.
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [scipy.misc.imrotate(img, angle) for img in clip]
        elif isinstance(clip[0], Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError(f'Expected numpy.ndarray or PIL.Image but got list of {type(clip[0])}')

        return rotated


class CenterCrop(object):
    """Extract center crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        clip (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped

class TenCrop(object):
    """
    Extract ten crops (center and corners) from a list of images, including their horizontal flips.

    Args:
        size (sequence or int): Desired output size for the crop in the format (h, w).
                                If size is a single integer, the output size will be (size, size).

    Attributes:
        size (tuple): The size of the crop in the format (h, w).

    Methods:
        __call__(clip):
            Extracts ten crops from the input clip, including their horizontal flips.

    Example:
        >>> transform = TenCrop(size=(224, 224))
        >>> cropped_clip = transform(clip)

    Raises:
        TypeError: If the input clip is not a list of numpy.ndarray or PIL.Image.
        ValueError: If the crop size is larger than the input image size.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (list of PIL.Image or numpy.ndarray): List of images to be cropped
                                                       in the format (h, w, c) for numpy.ndarray.

        Returns:
            list of PIL.Image or numpy.ndarray: List of cropped images, including their horizontal flips.

        Raises:
            TypeError: If the input clip is not a list of numpy.ndarray or PIL.Image.
            ValueError: If the crop size is larger than the input image size.
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))

        if w > im_w or h > im_h:
            raise ValueError('Initial image size should be larger than cropped size but got cropped sizes : '
                             '({w}, {h}) while initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h, w=w, h=h))

        # Create horizontally flipped version of the clip
        if isinstance(clip[0], np.ndarray):
            flip_clip = [np.fliplr(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            flip_clip = [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))

        # Calculate the coordinates for the center and corner crops
        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))

        all_x = [x1, 0, im_w - w, 0, im_w - w]
        all_y = [y1, 0, 0, im_h - h, im_h - h]

        # Extract the crops
        cropped = [F.crop_clip(clip, y, x, h, w) for x, y in zip(all_x, all_y)]
        flip_cropped = [F.crop_clip(flip_clip, y, x, h, w) for x, y in zip(all_x, all_y)]

        # Combine the original and flipped crops
        cropped.extend(flip_cropped)

        return cropped


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image

        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip
