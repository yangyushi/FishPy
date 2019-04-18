#!/usr/bin/env python
import numpy as np
from scipy.cluster import vq
import matplotlib.pyplot as plt


def vanilla_pca(images: np.ndarray) -> tuple:
    """
    :param images: collection of nd images
    :return: the projection matrix
    """
    image_num, dimension = images.shape
    mean = images.mean(axis=0)
    normalised = images - mean
    if dimension > image_num:
        covar = normalised @ normalised.T
        e, vh = np.linalg.eigh(covar)
        e[e < 0] = 0
        tmp = (normalised.T @ vh).T
        projection = tmp[::-1]
        variance = np.sqrt(e)[::-1]
        for i in range(projection.shape[1]):
            projection[:, i] /= variance
    else:
        u, variance, vh = np.linalg.svd(normalised)
        projection = vh[:image_num]
    return projection, variance, mean


def plot_pca(dim, mean, pcs, name='pca'):
    fig, axs = plt.subplots(2, 4)
    for i, ax in enumerate(axs.ravel()):
        if i == 0:
            ax.set_title('mean')
            ax.imshow(mean.reshape(dim, dim), cmap='bwr')
        else:
            p = pcs[i-1]
            ax.imshow(p.reshape(dim, dim), cmap='bwr')
            ax.set_title(f'#{i}')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.set_size_inches(12, 8)
    plt.savefig('pca.pdf')
    plt.close()


def get_kernels(images: np.ndarray, indices: np.ndarray, cluster_num: int, plot=True) -> list:
    """
    1. calculate the principle component of different images
    2. project images on some principles
    3. cluster the projected images
    4. calculate average for each cluster
    5. return the averaged images

    :param shapes: different images obtained from measure_shape.get_shapes
    :param indices: indices of the principle components
    :param cluster_num  : the number of clusters (k)
    """
    number, dim, dim = images.shape

    for_pca = np.reshape(images, (number, dim * dim))
    pcs, variance, mean = vanilla_pca(for_pca)

    if plot:
        plot_pca(dim, mean, pcs)

    projected = np.array([pcs[indices] @ (img - img.mean()) for img in for_pca])
    features = vq.whiten(projected)
    centroids, distortion = vq.kmeans(features, cluster_num)

    code, distance = vq.vq(features, centroids)

    if plot:
        fig, ax = plt.subplots(1, cluster_num)

    shape_kernels = []

    for k in range(cluster_num):
        indices = np.where(code == k)[0]
        shape_images = images[indices]
        average = shape_images.mean(0)
        if plot:
            ax[k].imshow(average.reshape(dim, dim), vmin=0, vmax=images.max())
        kernel = average.reshape(dim, dim)# - np.mean(average)
        shape_kernels.append(kernel)

    if plot:
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
        plt.tight_layout()
        plt.gcf().set_size_inches(15, 3)
        plt.savefig('kernels.pdf')
        plt.close()

    return shape_kernels


if __name__ == "__main__":
    shapes = np.load('./fish_shape_collection_cam-1.npy')
    indices = np.arange(2, 4)
    cluster_num = 4
    kernels = get_kernels(shapes, indices, cluster_num)
    np.save('shape_kernels_cam-1', kernels)
