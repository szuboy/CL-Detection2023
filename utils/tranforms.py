# !/usr/bin/env python
# -*- coding:utf-8 -*-


import random
import numpy as np
from skimage import transform as sk_transform


def generate_2d_gaussian_heatmap(heatmap: np.ndarray, center: tuple, sigma=20, radius=50):
    """
    function to generate 2d gaussian heatmap
    :param heatmap: heatmap array | 传入进来赋值的高斯热图
    :param center: a tuple, like (x0, y0) | 中心的坐标
    :param sigma: gaussian distribution sigma value | 高斯分布的sigma值
    :param radius: gaussian distribution radius | 高斯分布考虑的半径范围
    :return: heatmap array
    """
    x0, y0 = center
    xx, yy = np.ogrid[-radius:radius + 1, -radius:radius + 1]

    # generate gaussian distribution
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    # valid range
    height, width = np.shape(heatmap)
    left, right = min(x0, radius), min(width - x0, radius + 1)
    top, bottom = min(y0, radius), min(height - y0, radius + 1)

    # assign operation
    masked_heatmap = heatmap[y0 - top:y0 + bottom, x0 - left:x0 + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    # the np.maximum function is used to avoid aliasing of multiple landmarks on the same heatmap
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    Args:
        output_size tuple: Desired output size. The output is matched to output_size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = int(self.output_size[0]), int(self.output_size[1])

        image = sk_transform.resize(image, (new_h, new_w), mode='constant', preserve_range=False)
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': image, 'landmarks': landmarks}


class RandomHorizontalFlip(object):
    """
    Flip randomly the image in a sample.
    Args:
        p float: the probability of using horizontal flip augmentation
    """

    def __init__(self, p):
        assert isinstance(p, float)
        self.prob = p

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        if random.random() < self.prob:
            _, w = image.shape[:2]
            landmarks[:, 0] = w - landmarks[:, 0]
            image = image[:, ::-1, :].copy()

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """
    Convert image array in sample to Tensors.
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # generate all landmarks' heatmap
        h, w = image.shape[:2]
        n_landmarks = np.shape(landmarks)[0]
        heatmap = np.zeros((n_landmarks, h, w))
        for i in range(n_landmarks):
            center = (int(landmarks[i, 0] + 0.5), int(landmarks[i, 1] + 0.5))
            heatmap[i, :, :] = generate_2d_gaussian_heatmap(heatmap[i, :, :], center, sigma=8, radius=20)

        # swap color axis because numpy image: H x W x C but torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return image, heatmap

