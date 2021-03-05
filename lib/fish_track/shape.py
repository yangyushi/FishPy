#!/usr/bin/env python3
import configparser
import numpy as np
from scipy import ndimage


def is_inside(position, radius, boundary):
    result = True
    for dim in range(len(boundary)):
        result *= (position[dim] - np.ceil(radius) > 0)
        result *= (position[dim] + np.ceil(radius) < boundary[dim])
    return result


def get_maxima(image: np.ndarray, threshold: float, window_size):
    """
    return the location of features in the image
    the result is (u, v), not (row, column)
    :return maxima: shape is (n, 2)
    """
    mmap = ndimage.grey_dilation(image, window_size) == image
    mmap *= image > threshold
    labels, _ = ndimage.label(mmap)
    maxima = ndimage.center_of_mass(image, labels=labels, index=range(1, labels.max()+1))
    if len(maxima) > 1:
        maxima = np.flip(np.array(maxima), axis=1)
    else:
        maxima = np.empty((0, 2), dtype=int)
    return maxima


def get_sub_image_box(position, radius, image_shape=None):
    for dim in range((len(position))):
        lower_boundary = int(np.ceil(position[dim] - np.ceil(radius)))
        upper_boundary = int(lower_boundary + 2 * np.ceil(radius))
        if not isinstance(image_shape, type(None)):
            lower_boundary = max(lower_boundary, 0)
            upper_boundary = min(upper_boundary, image_shape[dim])
        yield slice(lower_boundary, upper_boundary, None)


def get_sub_images(image, centres, max_radius):
    """
    only works for 2d image
    """
    int_maps = []
    for centre in np.flip(centres, axis=1):  # flip: using (row, colume) for the matrix
        if not is_inside(centre, max_radius, image.shape):
            continue
        int_map = np.zeros((int(2 * np.ceil(max_radius)), int(2 * np.ceil(max_radius))))
        sub_image_box = list(get_sub_image_box(centre, max_radius, image.shape))
        int_map = image[tuple(sub_image_box)]
        int_maps.append(int_map)
    return int_maps


def align_sub_image(sub_image, want_ar=False):
    """
    Align the image
    Everything is in 2d
    """
    points = np.array(sub_image.nonzero(), dtype=np.float64)
    centre = np.vstack(points.mean(1))
    covar = np.cov(points)
    u, s, vh = np.linalg.svd(covar)
    after_rotation = (vh.T @ (points - centre)) + np.vstack(np.array(sub_image.shape) / 2)
    result = np.zeros(sub_image.shape)

    for p, pr in zip(points.T, after_rotation.T):
        ip = tuple(p.astype(int))    # indices of points before rotation
        ipr = tuple(pr.astype(int))  # indices of points after rotation
        if (pr > np.array(result.shape)).any():
            continue
        if result[ipr] == 0:
            result[ipr] = sub_image[ip]
        else:
            result[ipr] = max([sub_image[ip], result[ipr]])

    result = ndimage.grey_closing(result, 2)  # fill the interier holes

    if want_ar:
        return result, (s.max() / s.min())
    else:
        return result


def get_shapes(image, fish, report=False):
    """
    measure and aligh individual shapes in an image
    if report were True, also return the volumes and aspect_ratios
    """
    threshold = fish.threshold * image.max()
    window_size = fish.size_max

    fg = image * (image > threshold)

    maxima = get_maxima(image, threshold, window_size)

    if report:
        volumes = []
        aspect_ratios = []

    if len(maxima) == 0:
        shapes = []
    else:
        sub_images = get_sub_images(fg, maxima, window_size)
        shapes = []
        for i, sub_img in enumerate(sub_images):
            volume = np.sum(sub_img)
            is_similiar = True
            is_similiar *= np.sum(sub_img > 0) > fish.size_max
            is_similiar *= ndimage.label(sub_img > 0)[1] < 2
            is_similiar *= volume > fish.volume_min
            is_similiar *= np.sum(sub_img) < fish.volume_max

            if report:
                volumes.append(volume)

            if not is_similiar:
                continue

            aligned_image, aspect_ratio = align_sub_image(sub_img, want_ar=True)

            not_too_fat = aspect_ratio > fish.aspect_ratio_min
            not_too_slim = aspect_ratio < fish.aspect_ratio_max
            if not_too_fat and not_too_slim:
                shapes.append(aligned_image)
            if report:
                aspect_ratios.append(aspect_ratio)
    if report:
        return shapes, volumes, aspect_ratios
    else:
        return shapes
