#!/usr/bin/env python3
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage


def get_centres(img):
    """
    Get homogenerous coordinates of centres of disconnected part
    The algorithm won't work with overlapping objects
    The coordinate in an 2D image:
        ----------------------------->  X
        |[0, 0], [0, 1], ..., [0, W]
        |[1, 0], [1, 1], ..., [1, W]
        |                ...
        |[H, 0], [H, 1], ..., [H, W]
        |
        ∨ Y
        W: width, H: height
        notice you have to swap ij to get right XY:
    """
    labels = scipy.ndimage.label(img.sum(-1))[0]
    centres = []
    for i in range(4):
        i += 1
        centre_ij = (np.mean(np.where(labels == i), -1))  # index: i, j
        centre_xy = np.array([centre_ij[1], centre_ij[0]])  # ij ---> xy
        centres.append(centre_xy)
    return np.hstack([centres, np.ones((4, 1))])


def get_fundamental_from_projections(p1, p2):
    """
    p1: projection matrix of camera for image 1, shape (3, 4)
    p2: projection matrix of camera for image 2, shape (3, 4)
    F: fundamental matrix, shape (3, 3)
    image 1 ---> image 2
    point_1' * F * point_2 = 0
    """
    X = [None, None, None]
    X[0] = np.vstack([p1[1], p1[2]])
    X[1] = np.vstack([p1[2], p1[0]])
    X[2] = np.vstack([p1[0], p1[1]])

    Y = [None, None, None]
    Y[0] = np.vstack([p2[1], p2[2]])
    Y[1] = np.vstack([p2[2], p2[0]])
    Y[2] = np.vstack([p2[0], p2[1]])

    F = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            XY = np.vstack([X[i], Y[j]])
            F[i, j] = np.linalg.det(XY)
    return F


def multi_view_link(c1, c2, f):
    """
    c1: homogenerous centres in image 1, shape (N, 3)
    c2: homogenerous centres in image 2, shape (N, 3)
    supposing centres in image 1 is in the right order (order of c1)
    c2[order] gives the right order of objects in image 2
    """
    order = []
    for c in c1:
        line = c.dot(f)  # another center
        dist = [
                abs((line[0]*c[0] + line[1]*c[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2))
                for c in c2
                ]
        close_index = np.argmin(dist, axis=0)
        order.append(close_index)
    return order


def line2func(line):
    slope = - line[0] / line[1]
    intercept = - line[2] / line[1]
    return lambda x: slope * x + intercept



if __name__ == '__main__':
    from camera import view_1, view_2, view_3

    im1 = np.array(Image.open('particle_view1.png'))
    im2 = np.array(Image.open('particle_view2.png'))
    im3 = np.array(Image.open('particle_view3.png'))

    centres_1 = get_centres(im1)
    centres_2 = get_centres(im2)
    centres_3 = get_centres(im3)


    p1 = view_1.get_projection_matrix()
    p2 = view_2.get_projection_matrix()
    p3 = view_3.get_projection_matrix()

    f12 = get_fundamental_from_projections(p1, p2)
    f13 = get_fundamental_from_projections(p1, p3)
    o12 = multi_view_link(centres_1, centres_2, f12)
    o13 = multi_view_link(centres_1, centres_3, f13)

    centres_2 = centres_2[o12]
    centres_3 = centres_3[o13]

    x = np.linspace(0, 1000, 100)

    plt.imshow(im1)
    for c1 in centres_1:
        plt.scatter(c1[0], c1[1], marker='o', s=40, edgecolor='k', linewidth=1)
        plt.xticks([])
        plt.yticks([])
    plt.show()

    plt.imshow(im2)
    for c1, c2 in zip(centres_1, centres_2):
        lf2 = line2func(c1.dot(f12))
        plt.plot(x, lf2(x))
        plt.scatter(c2[0], c2[1], marker='o', s=40, edgecolor='k', linewidth=1)
        plt.xlim(0, im3.shape[1])
        plt.ylim(im3.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
    plt.show()

    plt.imshow(im3)
    for c1, c3 in zip(centres_1, centres_3):
        lf3 = line2func(c1.dot(f13))
        plt.plot(x, lf3(x))
        plt.scatter(c3[0], c3[1], marker='o', s=40, edgecolor='k', linewidth=1)
        plt.xlim(0, im3.shape[1])
        plt.ylim(im3.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
    plt.show()
