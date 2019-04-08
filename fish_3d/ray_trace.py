#!/usr/bin/env python3
import numpy as np
from scipy import optimize


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


def pl_dist(point, line):
    """distance between a point and a line"""
    ac = point - line['point']
    ab = line['unit']
    return np.linalg.norm(np.cross(ac, ab)) / np.linalg.norm(ab)


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


def is_inside_image(xy, image):
    x0, x1 = 0, image.shape[0]
    y0, y1 = 0, image.shape[1]
    x, y = xy
    is_inside = True
    is_inside *= x > x0
    is_inside *= x < x1
    is_inside *= y > y0
    is_inside *= y < y1
    return is_inside


def find_u(u, n, d, x, z):
    """
    The meanings of variable names can be found in this paper
        10.1109/CRV.2011.26
    """
    return (n**2 * (d**2 + u**2) - u**2) * (x - u)**2 - u**2 * z**2


def epipolar_refractive(xy, camera_1, camera_2, image_2, interface=0, normal=(0, 0, 1), step=1, n=1.33):
    """
    The meanings of variable names can be found in this paper
        10.1109/CRV.2011.26

    :param xy: (x, y) location of a pixel on image from camera_1, NOT (u, v)
    :param interface: height of water level in WORLD coordinate system
    :param normal: normal of the air-water interface, from water to air
    :param n: refractive index of water (or some other media)
    
    Here the goal is: 
        1.  For given pixel (xy) in image taken by camera #1, calculated
            it's projection on air-water interface (poi_1), and the
            direction of the refractted ray (trans_vec)
        2.  While the projection is still on image from camera_2:
                i.   Calculated the position (M) from poi_1 going along trans_vec
                ii.  Project M onto camera #2
                iii. Collect the projection points
    """
    
    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    poi_1 = get_poi(camera_1.p, interface, xy[::-1])
    incid = poi_1 - co_1
    trans = get_trans_vec(incid, normal=normal)
    is_in_image = True
    m = poi_1.copy()
    epipolar_pixels = []
    while is_in_image:
        m += step * trans
        z = abs(m[-1])
        d = abs(co_2[-1] - interface)
        x = np.linalg.norm(m[:2] - co_2[:2])
        u = optimize.root_scalar(find_u, args=(n, d, x, z), x0=x/1.2, x1=x).root
        assert u <= x
        assert u >= 0
        o = np.hstack((co_2[:2], interface))
        oq_vec = np.hstack((m[:2], interface)) - o
        q = o + u * (oq_vec / np.linalg.norm(oq_vec))
        uv_2 = camera_2.p @ np.hstack((q, 1))
        uv_2 /= uv_2[-1]
        uv_2 = uv_2[:2]  # (u, v, w)  -> (u, v)
        xy_2 = uv_2[::-1]  # (u, v)  -> (x, y)
        is_in_image = is_inside_image(xy_2, image_2)
        if is_in_image:
            epipolar_pixels.append(xy_2)
    return epipolar_pixels


def ray_trace_refractive(centres, cameras, z=0, normal=(0, 0, 1), refractive_index=1.33):
    """
    :param centres: a list of homogenerous centres in different views
    :param cameras: a list of *calibrated* Camera objects
    :param z: the z-value of the refractive interface
    """
    camera_origins = [-camera.r.T @ camera.t for camera in cameras]
    pois = [
        get_poi(camera.p, z, centre[::-1]) for camera, centre in zip(cameras, centres)
    ]  # ::-1 changes from (x, y) to (u, v)
    incid_rays = [poi - co for poi, co in zip(pois, camera_origins)]
    trans_rays = [get_trans_vec(incid, normal=normal) for incid in incid_rays]
    trans_lines = [{'unit': trans, 'point': poi} for trans, poi in zip(trans_rays, pois)]
    
    point_3d = get_intersect_of_lines(trans_lines)
    
    error = 0
    for line in trans_lines:
        error += pl_dist(point_3d, line)
    return point_3d, error


if __name__ == "__main__":
    l1 = {'unit': (1.3, 1.3, -10), 'point': (-14.2, 17, -1)}
    l2 = {'unit': (12.1, -17.2, 1.1), 'point': (1, 1, 1)}
    l3 = {'unit': (19.2, 31.8, 3.5), 'point': (2.3, 4.1, 9.8)}
    l4 = {'unit': (4, 5, 6), 'point': (1, 2, 3)}
    lines = [l1, l2, l3, l4]
    result = get_intersect_of_lines(lines).ravel()
    assert np.allclose(result, [-3.557, 7.736, 4.895], atol=0.001)
    assert np.allclose(get_trans_vec((0, 0, -3)), np.array([0, 0, -1]), atol=0.01)
