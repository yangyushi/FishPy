#!/usr/bin/env python3
import numpy as np


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
