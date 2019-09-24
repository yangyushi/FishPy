from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from . import ray_trace

dpi = 150


def see_corners(image_file, corner_number=(23, 15)):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, corner_number,
            flags=sum((
                cv2.CALIB_CB_FAST_CHECK,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                )))
    corners = np.squeeze(corners)
    plt.imshow(np.array(gray), cmap='gray')
    plt.plot(*corners.T[:2], color='tomato')
    plt.xlim(corners[:, 0].min() - 100, corners[:, 0].max() + 100)
    plt.ylim(corners[:, 1].min() - 100, corners[:, 1].max() + 100)
    plt.scatter(*corners[0].T, color='tomato')
    plt.axis('off')
    plt.show()


def plot_reproject(image, roi, features, pos_3d, camera, filename=None, water_level=0, normal=(0, 0, 1)):
    fig = plt.figure(figsize=(image.shape[1]/dpi, image.shape[0]/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    for point in pos_3d:
        color = (np.random.random(3) + 1) / 2
        xy = ray_trace.reproject_refractive(point, camera)
        ax.scatter(*xy, color='tomato', marker='+', lw=1, s=100)

    ax.scatter(features[0] + roi[1].start, features[1] + roi[0].start, color='tomato', facecolor='none')

    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.gcf().set_frameon(False)
    plt.gcf().axes[0].get_xaxis().set_visible(False)
    plt.gcf().axes[0].get_yaxis().set_visible(False)
    plt.axis('off')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_clusters(image, threshold, min_size, roi):
    """
    apply threshold to image and label disconnected part
    small labels (size < min_size) were erased
    the coordinates of labels were returned, shape: (label_number, 3)
    """
    binary = image > (image[roi].max() * threshold)
    mask = np.zeros(image.shape, dtype=int)
    mask[roi] = 1
    labels, _ = ndimage.label(binary)
    labels *= mask
    counted = np.bincount(labels.ravel())
    noise_indice = np.where(counted < min_size)
    mask = np.in1d(labels, noise_indice).reshape(image.shape)
    labels[mask] = 0
    labels, _ = ndimage.label(labels)
    clusters = []
    for value in np.unique(labels.ravel()):
        if value > 0:
            cluster = np.array(np.where(labels == value)).T
            clusters.append(cluster)
    return clusters


def plot_epl(line: List[float], image: np.ndarray):
    """
    Get the scatters of an epipolar line *inside* an image
    The images is considered as being stored in the (row [y], column [x]) manner
    """
    u = np.linspace(0, image.shape[1], 1000)
    v = -(line[2] + line[0] * u) / line[1]
    mask = np.ones(len(u), dtype=bool)
    mask[v < 0] = False
    mask[v >= image.shape[0]] = False
    uv = np.vstack([u, v]).T[mask]
    uv_dst = uv
    mask = np.ones(len(uv_dst), dtype=bool)
    mask[uv_dst[:, 0] < 0] = False
    mask[uv_dst[:, 0] >= image.shape[1]] = False
    mask[uv_dst[:, 1] < 0] = False
    mask[uv_dst[:, 1] >= image.shape[0]] = False
    epl = uv_dst[mask]
    return epl.T
