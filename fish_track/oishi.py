#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import signal
from scipy.spatial.distance import pdist, squareform, cdist


def join_pairs(pairs) -> list:
    if len(pairs) == 0:
        return []
    max_val = np.max(np.hstack(pairs)) + 1
    canvas = np.zeros((max_val, max_val), dtype=int)
    p = np.array(pairs)
    canvas[tuple(p.T)] = 1
    labels, _ = ndimage.label(canvas)
    joined_pairs = []
    for val in set(labels[labels > 0]):
        joined_pair = np.unique(np.vstack(np.where(labels == val)))
        joined_pairs.append(joined_pair)
    return joined_pairs


def rotate_kernel(kernel, angle) -> np.ndarray:
    """
    rotate the kernel with given angle
    and shift the highest value to the centre
    """
    after_rot = ndimage.rotate(kernel, angle, mode='constant', cval=kernel.min())
    centre = np.array(after_rot.shape) // 2
    maximum = np.array(np.unravel_index(np.argmax(after_rot), after_rot.shape))
    shift = centre - maximum
    shifted = ndimage.shift(after_rot, shift, mode='constant', cval=kernel.min())
    return get_sub_image(after_rot, centre, kernel.shape)


def get_sub_image(image, centre, window):
    win_x, win_y = window
    region_x = slice(centre[0] - win_x//2, centre[0] + win_x//2 + win_x%2)
    region_y = slice(centre[1] - win_y//2, centre[1] + win_y//2 + win_y%2)
    sub_image = image[region_x, region_y]
    return sub_image


def get_box_for_kernel(kernel, u, v, image) -> tuple:
    """
    get a boundary of the kernel in a bigger image
    the kernel is located at (u, v) in the image
    """
    size_u, size_v = image.shape
    top = kernel.shape[0] // 2
    bottom = kernel.shape[0] // 2 + kernel.shape[0] % 2
    right = kernel.shape[1] // 2
    left = kernel.shape[1] // 2 + kernel.shape[1] % 2

    # box don't go beyond image
    top = min(size_u - u, top)
    bottom = min(u, bottom)

    left = min(v, left)
    right = min(size_v - v, right)

    # box in kernel
    kcu, kcv = kernel.shape[0] // 2 + kernel.shape[0] % 2, kernel.shape[1] // 2 + kernel.shape[1] % 2  # kernel centre


    box_in_image = (slice(u - bottom, u + top), slice(v - left, v + right))
    box_in_kernel = (slice(kcu - bottom, kcu + top), slice(kcv - left, kcv + right))

    return box_in_image, box_in_kernel


def get_clusters(feature, image, kernels, angles, roi, kernel_threshold=0.0) -> list:
    """
    image: the processed binary image
    return the pixels in the image that 'belong' to different features
    the returnned results are (x, y), the shape is (number, dimension)
    """
    clusters = []
    image_in_roi = image[roi]

    for x, y, o, s, _, p in zip(*feature.tolist()):
        x, y, o, s = tuple(map(int, (x, y, o, s)))
        kernel = kernels[int(s)]
        kernel = rotate_kernel(kernel, angles[int(o)])

        box_in_image, box_in_kernel = get_box_for_kernel(kernel, y, x, image_in_roi)

        mask = (kernel > (kernel_threshold * kernel.max()))[box_in_kernel]

        sub_image = image_in_roi[box_in_image]

        offset_1 = np.array([b.start for b in box_in_image])
        offset_2 = np.array([r.start for r in roi])
        offset = np.hstack(offset_1 + offset_2)

        cluster_mi = np.array(np.nonzero(sub_image * mask)).T  # matrix indices, shape (n, 2)

        if len(cluster_mi) > 1:
            clusters.append(
                np.flip(cluster_mi + offset, axis=1)  # CONVERT (row, column) to (x, y)
            )

    return clusters


def o2v(orientation, angles) -> np.ndarray:
    """
    convert orientation to a unit vector
    the angles were in the unit of degree
    """
    assert orientation < len(angles), "angle array and orientation does not match"
    angle = angles[orientation]
    rad = angle / 180 * np.pi
    x, y = np.cos(rad), np.sin(rad)
    return np.array((x, y))


def get_oishi_kernels(kernels, rot_num=35):
    oishi_kernels = np.empty((len(kernels), rot_num, kernels[0].shape[0], kernels[0].shape[1]))
    for i, k in enumerate(kernels):
        for r in range(rot_num):
            kernel = rotate_kernel(k, angle=r*180/rot_num)
            oishi_kernels[i, r] = (kernel - kernel.mean()) / kernel.std()
    return oishi_kernels


def get_oishi_features(image, oishi_kernels, threshold=0.5, local_size=4):
    """
    :return : oishi features, (6, n) array with [x, y, orientation, shape, brightness, likelihood]
              the returned feature location is (x, y) in the image coordinate, NOT (row, colume) indices of the image matrix
    """
    local_max = (ndimage.grey_dilation(image, local_size) == image) * (image > image.max() * threshold)
    raw_features = np.array(local_max.nonzero()).T  # shape (n, 2), in the format of (row, column)

    window = oishi_kernels[0][0].shape
    orientations = np.empty(len(raw_features))
    shapes = np.empty(len(raw_features))
    brightness = np.empty(len(raw_features))
    likelihood = np.empty(len(raw_features))

    for i, feature in enumerate(raw_features):
        sub_im = get_sub_image(image, feature, window)
        if sub_im.shape != window:
            likelihood[i] = 0
            shapes[i], orientations[i] = 0, 0
            continue
        sub_im = (sub_im - sub_im.mean())
        corr = np.abs(oishi_kernels * sub_im) * sub_im  # last term give hight wight to bright spots
        corr = corr.sum(-1).sum(-1)  # sum over all sub-image
        likelihood[i] = np.max(corr)
        brightness[i] = image[tuple(feature)]
        shapes[i], orientations[i] = np.unravel_index(np.argmax(corr), corr.shape)

    positions = np.flip(raw_features.T, axis=0).astype(int)  # shape (2, n), CONVERT from (row, colume) to (x, y)
    oishi_features = np.vstack(
        (positions, orientations, shapes, brightness, likelihood/likelihood.std())
    )
    return oishi_features


def get_align_map(features, rot_num):
    dist_o0 = cdist(np.vstack(features[2]), np.vstack(features[2]))
    dist_o1 = cdist(np.vstack(features[2]), np.vstack(features[2]-rot_num))
    dist_o2 = cdist(np.vstack(features[2]), np.vstack(features[2]+rot_num))
    align_map = np.min((dist_o0, dist_o1, dist_o2), 0)  # deal with the "PBC" of orientation, 0 = 2 pi
    align_map = align_map / rot_num * 180
    return align_map


def verify_pair(features, pair, rot_num, orient_threshold):
    p1, p2 = features.T[pair]
    o1, o2 = p1[2] / rot_num * 180, p2[2] / rot_num * 180
    oi = np.argmin([abs(o1-o2), abs(o1-o2-180), abs(o1-o2+180)])
    o2 = (o2, o2+180, o2-180)[oi]
    o_mean = (o1 + o2) / 2
    shift = p1[:2] - p2[:2]
    shift_orient = np.arctan(shift[1] / shift[0]) / np.pi * 180
    if shift_orient < 0:
        shift_orient += 180
    if abs(shift_orient - o_mean) < orient_threshold * 2:
        return True
    else:
        return False



def refine_oishi_features(features, rot_num, dist_threshold, orient_threshold, likelihood_threshold, intensity_threshold):
    dist_xy = cdist(features[:2].T, features[:2].T)
    align_map = get_align_map(features, rot_num)
    is_close = dist_xy < dist_threshold
    is_aligned = align_map < orient_threshold
    is_same = np.triu(is_close * is_aligned, k=1)
    pairs_to_join = np.array(is_same.nonzero()).T
    #pairs_to_join = [pair for pair in pairs_to_join if verify_pair(features, pair, rot_num, orient_threshold)]
    pairs = join_pairs(pairs_to_join)
    to_del = []

    for p in pairs:
        to_keep = np.argmax(features[-1][p])
        to_del += np.concatenate((p[:to_keep], p[to_keep + 1:])).tolist()
    to_del = np.array(to_del)

    mask = np.ones(features.shape[1], dtype=bool)
    if len(to_del) > 0:
        mask[to_del] = False
    mask[features[-1] < likelihood_threshold] = False
    mask[features[-2] < (intensity_threshold * features[-2].max())] = False
    refined = features[:, mask]
    return refined
