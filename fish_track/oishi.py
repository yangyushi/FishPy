#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage 
from scipy import signal
import numba


@numba.jit(nopython=True, parallel=True)
def get_diff(image: np.ndarray, kernel: np.ndarray):
    lx, ly = image.shape
    lkx, lky = kernel.shape
    left = int(lkx // 2)
    top  = int(lky // 2)
    diff = np.zeros(image.shape)
    for x in range(left, lx - lkx + left):
        for y in range(top, ly - lky + top):
            error = 0
            for kx in range(lkx):
                for ky in range(lky):
                    error += np.abs(image[x - left + kx, y - top + ky] - kernel[kx, ky])
            diff[x, y] = error
    return diff
                    
def get_mse_nd(image: np.ndarray, angles: list, kernels:list) -> np.ndarray:
    mse = []
    for angle in angles:
        mse.append([])
        for kernel in kernels:
            area = kernel.shape[0] * kernel.shape[1]
            cc_kernel = ndimage.rotate(kernel, angle)
            diff = get_diff(image, cc_kernel) / area
            mse[-1].append(diff)
    return np.array(mse)

def rotate_kernel(kernel, angle):
    """
    rotate the kernel with given angle
    and shift the highest value to the centre
    """
    after_rot = ndimage.rotate(kernel, angle, mode='constant', cval=kernel.min())
    centre = np.array(after_rot.shape) // 2
    maximum = np.array(np.unravel_index(np.argmax(after_rot), after_rot.shape))
    shift = centre - maximum
    shifted = ndimage.shift(after_rot, shift, mode='constant', cval=kernel.min())
    return shifted


def get_cross_correlation_nd(image: np.ndarray, angles: list, kernels: list) -> np.ndarray:
    """
    rotate different kernels at different angles
    and calculate the correlations
    """
    corr = []
    cc_image = image
    for angle in angles:
        corr.append([])
        for kernel in kernels:
            cc_kernel = rotate_kernel(kernel, angle)
            centre = np.array(cc_kernel.shape) // 2
            roi = ( slice(centre[0] - kernel.shape[0] // 2, centre[0] + kernel.shape[0] // 2),
                    slice(centre[1] - kernel.shape[1] // 2, centre[1] + kernel.shape[1] // 2))
            cc_kernel = cc_kernel[roi]  # make sure all kernels have the same size
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
    maxima = mmap.nonzero()
    probabilities = mmap[maxima]

    return np.vstack([maxima, probabilities])


def oishi_locate_mse(image: np.ndarray, mse: np.ndarray, size=5, mse_threshold=20, img_threshold=0.2):
    """
    todo: write a MUCH faster version
    """
    num_angle, num_shape, num_x, num_y = mse.shape

    mmap = ndimage.minimum_filter(
            mse, size=[2 * num_angle, 2 * num_shape, 2 * size, 2 * size],
            ) == mse

    mmap *= mse < (mse.max() * mse_threshold)
    mmap *= image > (image.max() * img_threshold)
    maxima = np.array(mmap.nonzero())

    return maxima


def get_box_for_kernel(kernel, x, y):
    """
    get a boundary of the kernel in a bigger image
    the kernel is located at (x, y)
    """
    if kernel.shape[0] % 2:
        top = kernel.shape[0] // 2
        bottom = kernel.shape[0] // 2 + 1
    else:
        top = kernel.shape[0] // 2
        bottom = kernel.shape[0] // 2
    if kernel.shape[1] % 2:
        left = kernel.shape[1] // 2
        right = kernel.shape[1] // 2 + 1
    else:
        left = kernel.shape[1] // 2
        right = kernel.shape[1] // 2
    return slice(x - top, x + bottom), slice(y - left, y + right)


def get_clusters(feature, image, kernels, angles, roi, threshold=0.5):
    """
    return the pixels in the image that 'belong' to different features
    """
    clusters = []

    for o, s, x, y, p in zip(*feature.tolist()):
        kernel = kernels[s]
        kernel = rotate_kernel(kernel, angles[o])
        mask = kernel > 0
        box = get_box_for_kernel(kernel, x, y)

        fg = image[roi]
        binary = (fg > fg.max() * threshold)[box]

        offset_1 = np.array([b.start for b in box])
        offset_2 = np.array([r.start for r in roi])
        offset = np.hstack(offset_1 + offset_2)

        cluster = np.array(np.nonzero(binary * mask)).T
        clusters.append(cluster + offset)

    return clusters


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
