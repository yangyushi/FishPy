#!/usr/bin/env python
import numpy as np
from scipy import ndimage
from scipy.cluster import vq
import matplotlib.pyplot as plt


def vanilla_pca(images):
    """
    Args:
        images (np.ndarray): a collections of images to perform PCA

    Return:
        tuple: the projection matrix, the variance and mean
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


def add_shadow(x, sigma):
    """
    Effectively add a negative zone around image x where x > x.mean
    """
    mask = (ndimage.grey_dilation(x, 3) > 0) * (x < x.max() * 0.1)
    diff = ndimage.gaussian_filter(x, sigma)
    diff *= mask
    result = x - diff
    return result / result.max()


def get_kernels(images, indices, cluster_num, plot=True, sigma=0):
    """
    1. calculate the principle component of different images
    2. project images on some principles
    3. cluster the projected images
    4. calculate average for each cluster
    5. return the averaged images

    Args:
        images (np.ndarray): images obtained from :meth:`fish_track.shape.get_shapes`
        indices (np.ndarray): indices of the principle components
        cluster_num (int): the number of clusters (k)
        sigma (float): the sigma of the "shadow" add around the kernel
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
        kernel = average.reshape(dim, dim)  # - np.mean(average)
        #if sigma > 0:
        #    kernel = add_shadow(kernel, sigma)
        #kernel = kernel / np.sum(kernel > kernel.max() * 0.1)  # compensate for smaller kernels
        if plot:
            ax[k].imshow(kernel)
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
