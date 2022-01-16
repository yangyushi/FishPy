#!/usr/bin/env python
import numpy as np
from scipy import ndimage
from scipy.cluster import vq
import matplotlib.pyplot as plt


def pca(images):
    """
    Args:
        images (np.ndarray): a collections of images to perform PCA

    Return:
        tuple: the projection matrix, the variance and mean
    """
    image_num, dimension = images.shape
    mean = images.mean(axis=0)[np.newaxis, :]
    #std = images.std(axis=0)[np.newaxis, :]
    normalised = (images - mean)# / std  # (n, dim)
    covar = (normalised.T @ normalised) / image_num
    u, variance, vh = np.linalg.svd(covar)
    projection = vh[:image_num]
    return projection, variance, images.mean(axis=0)


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
    fig.set_size_inches(7, 4)
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.close()


def get_kernels(images, pc_indices, cluster_num, save_name=False, *args, **kwargs):
    """
    1. calculate the principle component of different images
    2. project images on some principles
    3. cluster the projected images
    4. calculate average for each cluster
    5. return the averaged images

    Args:
        images (np.ndarray): images obtained from \
            :meth:`fish_track.shape.get_shapes`
        pc_indices (np.ndarray): indices of the principle components
        cluster_num (int): the number of clusters (k)
        save_name (str): the name of the pca plot, if None there will be no plot
    """
    number, dim, dim = images.shape

    for_pca = np.reshape(images, (number, dim * dim))
    pcs, variance, mean = pca(for_pca)

    if save_name:
        plot_pca(dim, mean, pcs, name=f'{save_name}-eigen-fish.pdf')

    projected = np.array([pcs[pc_indices] @ (img - img.mean()) for img in for_pca])
    features = vq.whiten(projected)
    centroids, distortion = vq.kmeans(features, cluster_num)

    code, distance = vq.vq(features, centroids)

    if save_name:
        fig, ax = plt.subplots(1, cluster_num)

    shape_kernels = []

    for k in range(cluster_num):
        img_indices = np.where(code == k)[0]
        shape_images = images[img_indices]
        average = shape_images.mean(0)
        kernel = average.reshape(dim, dim)  # - np.mean(average)
        if save_name:
            ax[k].imshow(kernel)
        shape_kernels.append(kernel)

    if save_name:
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
        plt.gcf().set_size_inches(15, 3)
        plt.tight_layout()
        plt.savefig(f'{save_name}-kernels.pdf')
        plt.close()

    return shape_kernels


if __name__ == "__main__":
    shapes = np.load('./fish_shape_collection_cam-1.npy')
    indices = np.arange(2, 4)
    cluster_num = 4
    kernels = get_kernels(shapes, indices, cluster_num)
    np.save('shape_kernels_cam-1', kernels)
