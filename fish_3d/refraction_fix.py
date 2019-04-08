#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def get_poi(p, z, x):
    """
    - p: the projection matrix of the camera
    - z: the hight of water level with respect to world origin
    - x: the position (u, v) of object on the image, unit is pixel
    - poi, point on interface. See the sketch for its meaning

                   camera
                  /
                 /
                /
    ---------[poi]-------- air-water interface (z = z)
               |
               |
               |
             fish
    """
    M = np.zeros((2, 2))
    b = np.zeros((2, 1))
    for i in range(2):
        for j in range(2):
            M[i, j] = p[i, j] - x[i] * p[2, j]
        b[i] = x[i] * p[2, 2] * z + x[i] * p[2, 3] - p[i, 2] * z - p[i, 3]
    x, y = np.linalg.solve(M, b).ravel()
    return np.array([x, y, z])

    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(2):
            M[i, j] = p[i, j]# * x[j]
    M[2, 2] = -1

    b = np.zeros((3, 1))
    for i in range(2):
        b[i] = x[i] - p[i, 2] * z - p[i, 3]
    b[2] = - p[2, 2] * z - p[2, 3]

    x, y, w = np.linalg.solve(M, b).ravel()
    return np.array([x, y, z])


def get_trans_vec(incident_vec, refractive_index=1.33, normal=(0, 0, 1)):
    """
    get the unit vector of transmitted ray from air to water
    the refractive index of water is 1.33 @ 25°C
    the normal vector facing up, the coordinate is looks like

            ^ +z
    air     |
            | 
    ------------------>
            |
    water   |
            |
    """
    rri = 1 / refractive_index
    n = np.array(normal)
    i = np.array(incident_vec)
    i = i / np.linalg.norm(i)
    cos_i = -i @ n
    sin_t_2 = rri ** 2 * (1 - cos_i ** 2)
    t = rri * i + (rri * cos_i - np.sqrt(1 - sin_t_2)) * n
    return t

def get_intersect_of_lines(lines):
    """
    lines: a list containing many lines
    line: a dict containing the unit vector and a base point
    return the a point in 3D whose distances sum to all lines is minimum
    (I followed this answer: https://stackoverflow.com/a/48201730/4116538)
    """
    line_num = len(lines)
    M = np.zeros((3, 3))
    b = np.zeros((3, 1))
    for k, line in enumerate(lines):
        e = np.array(line['unit'])  # e --> (x, y, z)
        e = e / np.linalg.norm(e)  # normalise incase e is not unit vector
        p = np.array(line['point'])  # p --> (x, y, z)
        e2 = e @ e
        ep = e @ p
        for i in range(3):
            for j in range(3):
                M[i, j] += e[i] * e[j]
            M[i, i] -= e2
            b[i] += e[i] * ep - p[i] * e2
    return np.linalg.solve(M, b).ravel()

if __name__ == "__main__":
    l1 = {'unit': (1.3, 1.3, -10), 'point': (-14.2, 17, -1)}
    l2 = {'unit': (12.1, -17.2, 1.1), 'point': (1, 1, 1)}
    l3 = {'unit': (19.2, 31.8, 3.5), 'point': (2.3, 4.1, 9.8)}
    l4 = {'unit': (4, 5, 6), 'point': (1, 2, 3)}
    lines = [l1, l2, l3, l4]
    result = get_intersect_of_lines(lines).ravel()
    assert np.allclose(result, [-3.557, 7.736, 4.895], atol=0.001)
    assert np.allclose(get_trans_vec((0, 0, -3)), np.array([0, 0, -1]), atol=0.01)
