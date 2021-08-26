import re
import os
import numpy as np
import fish_track as ft
from scipy import ndimage

def get_updated_name(filename, name_map):
    """
    Transform raw filename into organised names like cam-1

    Args:
        filename (str): filename with the suffix
        name_map (dict): mapping from target to raw name
    """
    raw_name = os.path.splitext(filename)[0]

    if raw_name in name_map.keys():
        name = raw_name
    else:
        matched = False
        for target, raw in name_map.items():
            if re.match(raw, filename):
                name = target
                matched = True
        if not matched:
            return None
    return name


def get_shapes(image, threshold, win_size):
    """
    measure and aligh individual shapes in an image
    """
    threshold = threshold * image.max()

    fg = image * (image > threshold)

    maxima = ft.shape.get_maxima(image, threshold, win_size)

    if len(maxima) == 0:
        shapes = []
    else:
        sub_images = ft.shape.get_sub_images(fg, maxima, win_size)
        shapes = []
        for i, sub_img in enumerate(sub_images):
            is_connected = ndimage.label(sub_img > 0)[1] < 2
            not_small = np.sum(sub_img > 0) > win_size
            if is_connected and not_small:
                aligned_image, aspect_ratio = ft.shape.align_sub_image(
                    sub_img, want_ar=True
                )
                shapes.append(aligned_image)
    return shapes

