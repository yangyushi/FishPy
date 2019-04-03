#!/usr/bin/env python3
import numpy as np
from scipy import ndimage 
from scipy import signal


def get_cross_correlation_nd(image: np.ndarray, angles: list, kernels: list) -> np.ndarray:
    corr = []
    cc_image = image - image.mean()
    for angle in angles:
        corr.append([])
        for kernel in kernels:
            cc_kernel = ndimage.rotate(kernel, angle)
            cross_correlation = signal.correlate(cc_image, cc_kernel, mode='same')
            corr[-1].append(cross_correlation)
    return np.array(corr)


def oishi_locate(image: np.ndarray, cc: np.ndarray, size=5, cc_threshold=0.2, img_threshold=0.2):
    """
    orientation invariance and shape invariance location
    currently very computational inefficient
    but the snack from oishi is acturally good

    :param image: 2D numpy array for locating features
    :param cc:    cross-correlation of image with different kernels with different rotation
                  4D numpy array with shape (orientation )
    :param size:  Spatial region where 
    """
    num_angle, num_shape, num_x, num_y = cc.shape

    mmap = ndimage.maximum_filter(
            cc, size=[2 * num_angle, 2 * num_shape, 2 * size, 2 * size],
            ) == cc

    mmap *= cc > (cc.max() * cc_threshold)
    mmap *= image > (image.max() * img_threshold)
    maxima = np.array(mmap.nonzero())

    return maxima


if __name__ == "__main__":
    import load_video
    import matplotlib.pyplot as plt

    want_cc = True

    images = load_video.iter_video('../clip.mp4')
    background = np.load('background.npy')
    image = next(images)
    fg = background - image
    for_locating = ndimage.gaussian_filter(fg, 2)

    kernels = np.load('shape_kernels.npy')
    angles = np.linspace(0, 180, 8, endpoint=True)

    if want_cc:
        cross_correlation = get_cross_correlation_nd(for_locating, angles, kernels)
        np.save('cc', cross_correlation)
    else:
        cross_correlation = np.load('cc.npy')

    maxima = oishi_locate(for_locating, cross_correlation, cc_threshold=0.2, img_threshold=0.2, size=10)
    o, r, x, y = maxima

    length = 50

    for i, m in enumerate(maxima.T):
        angle = angles[o[i]] / 180 * np.pi
        base = m[2:].astype(np.float64)
        plt.plot(
            [base[1] - length/2 * np.sin(angle), base[1] + length/2 * np.sin(angle)],
            [base[0] - length/2 * np.cos(angle), base[0] + length/2 * np.cos(angle)],
            color='w', linewidth=0.5,
        )

    plt.imshow(image, cmap='gray')
    plt.scatter( y, x, color='red', edgecolor='w', marker='o', linewidth=0.5)
    plt.gcf().set_frameon(False)
    plt.axis('off')
    plt.gcf().axes[0].get_xaxis().set_visible(False)
    plt.gcf().axes[0].get_yaxis().set_visible(False)
    plt.savefig('oishi_locate.pdf')
    plt.close()
