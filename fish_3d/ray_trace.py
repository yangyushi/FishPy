#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy import optimize
from itertools import product
from typing import List, Tuple, Dict
import cv2

try:
    from . import cray_trace
    c_ext = True
except ImportError:
    c_ext = False



def get_intersect_of_lines_slow(lines):
    """
    lines: a list containing many lines
    line: a dict containing the unit vector and a base point
    return the a point in 3D whose distances sum to all lines is minimum
    (I followed this answer: https://stackoverflow.com/a/48201730/4116538)
    todo: no solve
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


def py_get_intersect_of_lines_batch(lines):
    """
    lines = [line, ...], shape -> (n, view, 2, 3)
    line = [points (a), unit directions (v)]
    M(3, 3) @ x(3, n) = b(3, n)
    """
    M = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,  ), dtype=np.float64)
    xyz = []
    for line in lines:
        M *= 0
        b *= 0
        for view in line:
            a, v = view
            v1, v2, v3 = v
            tmp = np.array((
                    (-v2**2 - v3**2, v1*v2, v1*v3),
                    (v1*v2, -v1**2 - v3**2, v2*v3),
                    (v1*v3, v2*v3, -v1**2 - v2**2)
                    ))
            M += tmp
            b += tmp @ a
        x = np.linalg.solve(M, b).ravel()
        xyz.append(x)
    return np.array(xyz)

if c_ext:
    get_intersect_of_lines_batch = cray_trace.get_intersect_of_lines
else:
    get_intersect_of_lines_batch = py_get_intersect_of_lines_batch

def get_intersect_of_lines(lines):
    """
    lines = [line, ...], shape -> (n, 2, 3)
    line = [points (a), unit directions (v)]
    """
    M = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,  ), dtype=np.float64)
    for line in lines:
        a, (v1, v2, v3) = line
        tmp = np.array((
                (-v2**2 - v3**2, v1*v2, v1*v3),
                (v1*v2, -v1**2 - v3**2, v2*v3),
                (v1*v3, v2*v3, -v1**2 - v2**2)
                ))
        M += tmp
        b += tmp @ a
    return np.linalg.solve(M, b).ravel()


def pl_dist(point, line):
    """
    distance between a point and a line
    todo: write cross product explicitly
    """
    ac = point - line['point']
    ab = line['unit']
    return np.linalg.norm(np.cross(ac, ab)) / np.linalg.norm(ab)


def pl_dist_faster(point, lines):
    """
    point -> (3,)
    line -> [point, unit_vector] (2, 3)
    """
    delta = np.array([point]) - lines[:, 0, :]  # (n, 3)
    return np.linalg.norm(np.cross(delta, lines[:, 1, :]))


def pl_dist_batch(points, lines):
    """
    points -> (n, 3,)
    lines -> (n, view, 2, dim) [dim = 3]
    """
    view_num = lines.shape[1]
    dists = 0
    for v in range(view_num):
        delta = points - lines[:, v, 0, :]  # (num, 0)
        d = np.linalg.norm(np.cross(delta, lines[:, v, 1, :]), axis=-1)
        dists += d
    #dists = np.mean(dists, 0)  # (view, n, dim) -> (n, dim)
    return dists / view_num


def get_poi(camera: 'Camera', z: float, coordinate: np.ndarray):
    """
    - camera: a Camera instance
    - z: the hight of water level with respect to world origin
    - corrdinate: the position (u, v) of object on the image, unit is pixel
        it can be one point or many points, but they should be undistorted
    - poi, point on interface. See the sketch for its meaning
    - return [X, Y, Z], shape (3, n)

                   camera
                  /
                 /
                /
    ---------[poi]-------- air-water interface (z = z)
               |
               |
               |
             fish

    The equation: P @ [x, y, z, 1]' = [c * v, c * u, c]' are solved for x, y, c, knowing everything else
    """
    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = camera.p.ravel()

    if coordinate.ndim == 1:  # (just one point)
        x, y = coordinate
        z = z
    elif coordinate.ndim == 2:  # (number, dim)
        x, y = coordinate.T
        z = z * np.ones(x.shape)
    else:
        raise RuntimeError("The shape of coordinates array is not valid")

    X =  (z*p12*p23 - z*p12*p33*y - z*p13*p22 + z*p13*p32*y + z*p22*p33*x - \
          z*p23*p32*x + p12*p24 - p12*p34*y - p14*p22 + p14*p32*y + p22*p34*x - p24*p32*x) /\
         (p11*p22 - p11*p32*y - p12*p21 + p12*p31*y + p21*p32*x - p22*p31*x)

    Y = -(z*p11*p23 - z*p11*p33*y - z*p13*p21 + z*p13*p31*y + z*p21*p33*x - \
          z*p23*p31*x + p11*p24 - p11*p34*y - p14*p21 + p14*p31*y + p21*p34*x - p24*p31*x) /\
         (p11*p22 - p11*p32*y - p12*p21 + p12*p31*y + p21*p32*x - p22*p31*x)

    return np.array([X, Y, z])


def get_poi_cluster(camera, z, coordinate):
    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = camera.p.ravel()
    x, y = coordinate.T
    z = z * np.ones(x.shape)
    X =  (z*p12*p23 - z*p12*p33*y - z*p13*p22 + z*p13*p32*y + z*p22*p33*x - \
          z*p23*p32*x + p12*p24 - p12*p34*y - p14*p22 + p14*p32*y + p22*p34*x - p24*p32*x) /\
         (p11*p22 - p11*p32*y - p12*p21 + p12*p31*y + p21*p32*x - p22*p31*x)

    Y = -(z*p11*p23 - z*p11*p33*y - z*p13*p21 + z*p13*p31*y + z*p21*p33*x - \
          z*p23*p31*x + p11*p24 - p11*p34*y - p14*p21 + p14*p31*y + p21*p34*x - p24*p31*x) /\
         (p11*p22 - p11*p32*y - p12*p21 + p12*p31*y + p21*p32*x - p22*p31*x)

    return np.array([X, Y, z])


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
    i = incident_vec / np.linalg.norm(incident_vec)
    cos_i = -i @ n
    sin_t_2 = rri ** 2 * (1 - cos_i ** 2)
    t = rri * i + (rri * cos_i - np.sqrt(1 - sin_t_2)) * n
    return t / np.linalg.norm(t)


def get_trans_vecs(incident_vecs, refractive_index=1.33, normal=(0, 0, 1)):
    """
    get the *unit vector* of transmitted ray from air to water
    the refractive index of water is 1.33 @ 25°C
    the normal vector facing up, the coordinate is looks like
    :param incident_vecs: vectors representing many incident rays, shape (n, 3)

            ^ +z
    air     |
            |
    ------------------>
            |
    water   |
            |

    """
    rri = 1 / refractive_index
    n = np.array([normal]).T  # (3, 1)
    i = incident_vecs / np.linalg.norm(incident_vecs, axis=1, keepdims=True)  # (n, 3)
    cos_i = -i @ n  # (n, 1)
    sin_t_2 = rri ** 2 * (1 - cos_i ** 2)  # (n, 1)
    t = rri * i + (rri * cos_i - np.sqrt(1 - sin_t_2)) * n.T
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def is_inside_image(uv, image):
    row_0, row_1 = 0, image.shape[0]
    col_0, col_1 = 0, image.shape[1]
    col, row = uv
    is_inside = True
    is_inside *= row > row_0
    is_inside *= row < row_1
    is_inside *= col > col_0
    is_inside *= col < col_1
    return is_inside


def find_u(u, n, d, x, z):
    """
    The meanings of variable names can be found in this paper
        10.1109/CRV.2011.26
    """
    return (n**2 * (d**2 + u**2) - u**2) * (x - u)**2 - u**2 * z**2


def epipolar_refractive(uv, camera_1, camera_2, image_2, interface=0, normal=(0, 0, 1), step=1, n=1.33):
    """
    The meanings of variable names can be found in this paper
        10.1109/CRV.2011.26

    :param uv: (u, v) location of a pixel on image from camera_1, NOT (x, y)
    :param interface: height of water level in WORLD coordinate system
    :param normal: normal of the air-water interface, from water to air
    :param n: refractive index of water (or some other media)

    Here the goal is:
        1.  For given pixel (u, v) in image taken by camera #1, calculated
            it's projection on air-water interface (poi_1), and the
            direction of the refractted ray (trans_vec)
        2.  While the projection is still on image from camera_2:
                i.   Calculated the position (M) from poi_1 going along trans_vec
                ii.  Project M onto camera #2
                iii. Collect the projection points
    """

    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    poi_1 = get_poi(camera_1, interface, uv)
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
        is_in_image = is_inside_image(uv_2, image_2)
        if is_in_image:
            epipolar_pixels.append(uv_2)
    return epipolar_pixels


def epipolar_draw(uv, camera_1, camera_2, image_2, interface=0, depth=400, normal=(0, 0, 1), n=1.33):
    """
    return all the pixels that contains the epipolar line in image_2, re-projected & distorted

    The meanings of variable names can be found in this paper

    :param uv: (u, v) location of a pixel on image from camera_1
    :param interface: height of water level in WORLD coordinate system
    :param normal: normal of the air-water interface, from water to air
    :param n: refractive index of water (or some other media)
    """
    poi_1 = get_poi(camera_1, interface, np.array(uv))
    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    incid = poi_1 - co_1
    trans = get_trans_vec(incid, normal=normal)
    is_in_image = True
    m = poi_1.copy()
    epipolar_pixels = []
    step = 1
    count = 0
    last_uv = None

    while is_in_image:
        m += step * trans
        z = abs(m[-1])
        if z > depth:
            break
        d = abs(co_2[-1] - interface)
        x = np.linalg.norm(m[:2] - co_2[:2])
        u = optimize.root_scalar(find_u, args=(n, d, x, z), x0=x*0.8, x1=x*0.5).root
        if (u > x) or (u < 0):
            print("root finding in refractive epipolar geometry fails")
            break
        o = np.hstack((co_2[:2], interface))
        oq_vec = np.hstack((m[:2], interface)) - o
        q = o + u * (oq_vec / np.linalg.norm(oq_vec))
        uv_2, _ = cv2.projectPoints(
                objectPoints=np.vstack(q).T,
                rvec=camera_2.rotation.as_rotvec(),
                tvec=camera_2.t,
                cameraMatrix=camera_2.k,
                distCoeffs=camera_2.distortion
        )
        uv_2 = np.ravel(uv_2).astype(int)

        is_in_image = is_inside_image(uv_2, image_2)

        if count == 0:
            is_connected = True
            is_different = True
        else:
            is_connected = np.sum(np.power(uv_2 - last_uv, 2)) <= 2
            is_different = not np.allclose(uv_2, last_uv)

        should_draw = is_different and is_in_image and is_connected

        if should_draw:
            epipolar_pixels.append(uv_2.copy())
            epipolar_pixels.append(uv_2.copy() + 1)
            last_uv = uv_2.copy()
        else:
            if not is_different:
                step *= 2
            if not is_connected:
                m -= step * trans
                step /= 3
        count += 1

    return np.array(epipolar_pixels)


def epipolar_la(uv, camera_1, camera_2, image_2, interface=0, depth=400, normal=(0, 0, 1), n=1.33):
    """
    linear approximation for epipolar line under water
    the line path through two UNDISTORTED projection points, one at the interface one below water
    use 5 epipolar points under water and do a linear fit
    """
    poi_1 = get_poi(camera_1, interface, np.array(uv))  # uv should be undistorted
    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    incid = poi_1 - co_1
    trans = get_trans_vec(incid, normal=normal)
    X = np.zeros((6, 2))
    Y = np.zeros((6, 1))
    ray_length = abs(depth / trans[-1])
    for i, step in enumerate([ray_length/4, ray_length/2, ray_length*3/4, ray_length, ray_length*2]):
        m = poi_1 + step * trans
        z = abs(m[-1] - interface)
        d = abs(co_2[-1] - interface)
        x = np.linalg.norm(m[:2] - co_2[:2])
        u = get_u(n, d, x, z)
        o = np.hstack((co_2[:2], interface))
        oq_vec = np.hstack((m[:2], interface)) - o
        q = o + u * (oq_vec / np.linalg.norm(oq_vec))
        uv_2 = camera_2.p @ np.hstack((q, 1))
        uv_2 = uv_2/uv_2[-1]
        X[i, :] = uv_2[0], 1
        Y[i, 0] = uv_2[1]
    a, b = (np.linalg.inv(X.T @ X) @ X.T) @ Y  # least square fit
    return a, b

def get_u(n, d, x, z):
    """
    n - relative refractive index
    d, x, z - length in mm
    """
    x = x / 1000
    d = d / 1000
    z = z / 1000

    A = n**2 - 1  # u^4
    B = - 2 * A * x # u^3
    C = d**2 * n**2 + A * x**2 - z**2  # u^2
    D = -2 * d**2 * n**2 * x  # u^1
    E = d**2 * n**2 * x**2

    p1 = 2*C**3 - 9*B*C*D + 27*A*D**2 + 27*B**2*E - 72*A*C*E;

    p2 = p1 + np.sqrt(-4 * (C**2 - 3*B*D + 12*A*E)**3 + p1**2)
    p3 = (C**2 - 3*B*D + 12*A*E) / (3*A * (p2/2)**(1/3)) +\
         ((p2/2)**(1/3)) / (3*A)
    p4 = np.sqrt( B**2 / (4*A**2) - (2*C)/(3*A) + p3 )
    p5 = B**2 / (2*A**2) - (4*C)/(3*A) - p3

    p6 = (-(B/A)**3 + (4*B*C)/A**2 - 8*D/A) / (4*p4)

    if (p5 - p6) > 0:
        u = -B / (4*A) - p4/2 - np.sqrt(p5 - p6)/2
        if (u > 0) and (u <= x):
            return u * 1000

        u = -B / (4 * A) - p4 / 2 + np.sqrt(p5 - p6) / 2
        if (u > 0) and (u <= x):
            return u * 1000

    if (p5 + p6) > 0:
        u = -B / (4 * A) + p4 / 2 - np.sqrt(p5 + p6) / 2
        if (u > 0) and (u <= x):
            return u * 1000

        u = -B / (4 * A) + p4 / 2 + np.sqrt(p5 + p6) / 2
        if (u > 0) and (u <= x):
            return u * 1000

    raise ValueError("Root finding for u failed, z = " % z)

def epipolar_la_draw(uv, camera_1, camera_2, image_2, interface=0, depth=400, normal=(0, 0, 1), n=1.33):
    """
    linear approximation for epipolar line under water
    use 3 epipolar points under water and do a linear fit
    """
    poi_1 = get_poi(camera_1, interface, np.array(uv))
    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    incid = poi_1 - co_1
    trans = get_trans_vec(incid, normal=normal)
    X = np.zeros((3, 2))
    Y = np.zeros((3, 1))
    ray_length = abs(depth / trans[-1])
    for i, step in enumerate([0, ray_length/2, ray_length]):
        m = poi_1 + step * trans
        z = abs(m[-1])
        d = abs(co_2[-1] - interface)
        x = np.linalg.norm(m[:2] - co_2[:2])
        u = optimize.root_scalar(find_u, args=(n, d, x, z), x0=x*0.8, x1=x*0.5).root
        if (u > x) or (u < 0):
            print("root finding in refractive epipolar geometry fails")
            break
        o = np.hstack((co_2[:2], interface))
        oq_vec = np.hstack((m[:2], interface)) - o
        q = o + u * (oq_vec / np.linalg.norm(oq_vec))
        uv_2 = camera_2.project(q)
        X[i, :] = uv_2[0], 1
        Y[i, 0] = uv_2[1]
    a, b = (np.linalg.inv(X.T @ X) @ X.T) @ Y  # least square fit
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y = a * x + b
    return np.array([x, y]).T


def ray_trace_refractive_cluster(clusters, cameras, z=0, normal=(0, 0, 1), refractive_index=1.33):
    camera_origins = [np.vstack(-camera.r.T @ camera.t) for camera in cameras]  # 3 coordinates whose shape is (3, 1)
    pois_mv = []
    for camera, cluster in zip(cameras, clusters):
        pois = get_poi_cluster(camera, z=z, coordinate=cluster).T
        pois_mv.append(pois)  # poi shape: (n, dim)
    incid_rays_mv = [poi - co.T for poi, co in zip(pois_mv, camera_origins)]
    trans_rays_mv = [
            np.stack([
                pois,
                get_trans_vecs(incid_rays, normal=normal)
                ], axis=1) for incid_rays, pois in zip(incid_rays_mv, pois_mv)
    ]  # (view, n, 2, dim)
    combinations = np.array(list(product(*trans_rays_mv)))  # shape: (n^view, view, 2, dim), can be HUGE!
    points_3d = get_intersect_of_lines_batch(combinations)
    error = pl_dist_batch(points_3d, combinations)
    return points_3d, error


def get_reproj_err(point_3d, points_2d, cameras, water_level, normal):
    reproj_err = 0
    for p_2d, cam in zip(points_2d, cameras):
        reproj = reproject_refractive_no_distort(point_3d, cam, water_level, normal)
        reproj_err += np.linalg.norm(reproj - p_2d)
    return reproj_err / len(cameras)


def ray_trace_refractive_trajectory(trajectories: List[np.ndarray], cameras: List['Camera'], z=0, normal=(0, 0, 1), refractive_index=1.33):
    """
    reconstruct a trajectory from many views
    :param trajectories: list of positions belonging to the same individual at different time points, shape (time_points, dim)
    """
    n_view, (n_time, n_dim) = len(trajectories), trajectories[0].shape
    camera_origins = [np.vstack(-camera.r.T @ camera.t) for camera in cameras]  # 3 coordinates whose shape is (3, 1)
    pois_mv = np.empty((n_view, n_time, 3))

    for i, (camera, traj) in enumerate(zip(cameras, trajectories)):
        poi = get_poi(camera, z=z, coordinate=traj).T
        pois_mv[i] = get_poi(camera, z=z, coordinate=traj).T # poi shape: (n_time, n_dim)

    incid_rays_mv = [poi - co.T for poi, co in zip(pois_mv, camera_origins)]

    trans_rays_mv = np.array([
            np.stack([
                pois,
                get_trans_vecs(incid_rays, normal=normal)
                ], axis=1) for incid_rays, pois in zip(incid_rays_mv, pois_mv)
    ])  # (n_view, n_time, 2, n_dim)
    trans_rays_mv = np.moveaxis(trans_rays_mv, 0, 1)
    points_3d = get_intersect_of_lines_batch(trans_rays_mv)
    error = pl_dist_batch(points_3d, trans_rays_mv)
    return points_3d, error


def ray_trace_refractive(centres, cameras, z=0, normal=(0, 0, 1), refractive_index=1.33):
    """
    :param centres: a list of centres (u, v) in different views, they should be undistorted
    :param cameras: a list of *calibrated* Camera objects
    :param z: the z-value of the refractive interface
    """
    camera_origins = [-camera.r.T @ camera.t for camera in cameras]
    pois = []
    for camera, centre in zip(cameras, centres):
        pois.append(get_poi(camera, z=z, coordinate=centre))
    incid_rays = [poi - co for poi, co in zip(pois, camera_origins)]
    trans_rays = [get_trans_vec(incid, normal=normal) for incid in incid_rays]
    trans_lines = [{'unit': t, 'point': poi} for t, poi in zip(trans_rays, pois)]

    point_3d = get_intersect_of_lines_slow(trans_lines)

    error = 0
    for line in trans_lines:
        error += pl_dist(point_3d, line)
    return point_3d, error / len(cameras)


def ray_trace_refractive_faster(centres, cameras, z=0, normal=(0, 0, 1), refractive_index=1.33):
    """
    :param centres: a list of centres (u, v) in different views, they should be undistorted
    :param cameras: a list of *calibrated* Camera objects
    :param z: the z-value of the refractive interface
    """
    camera_origins = [-camera.r.T @ camera.t for camera in cameras]
    pois = []
    for camera, centre in zip(cameras, centres):
        pois.append(get_poi(camera, z=z, coordinate=centre))
    incid_rays = np.array([poi - co for poi, co in zip(pois, camera_origins)])
    trans_rays = get_trans_vecs(incid_rays, normal=normal)
    trans_lines = np.array([[p, t] for p, t in zip(pois, trans_rays)])
    point_3d = get_intersect_of_lines(trans_lines)
    error = pl_dist_faster(point_3d, trans_lines)
    return point_3d, error / len(cameras)


def cost_snell(xy, z, location, origin, normal, refractive_index):
    poi = np.hstack([xy, z])
    incid = poi - origin
    trans = get_trans_vec(incid, refractive_index, normal)
    line = {'unit': trans, 'point': poi}
    distance = pl_dist(location, line)
    return distance


def same_direction(v1, v2, v3, axis):
    return (v1[axis] - v2[axis]) * (v3[axis] - v1[axis])


def reproject_refractive(xyz, camera, water_level=0, normal=(0, 0, 1), refractive_index=1.333):
    """
    variable names follwoing https://ieeexplore.ieee.org/document/5957554, figure 1
    """
    co = -camera.r.T @ camera.t
    d = co[-1] - water_level
    x = np.linalg.norm(co[:2] - xyz[:2])
    z = abs(xyz[-1] - water_level)
    u = get_u(refractive_index, d, x, z)
    o = np.hstack((co[:2], water_level))
    oq_vec = np.hstack((xyz[:2], water_level)) - o
    oq_vec /= np.linalg.norm(oq_vec)
    poi = o + u * oq_vec
    uv, _ = cv2.projectPoints(
            objectPoints=np.vstack(poi).T,
            rvec=camera.rotation.as_rotvec(),
            tvec=camera.t,
            cameraMatrix=camera.k,
            distCoeffs=camera.distortion
    )
    return uv.ravel()


def reproject_refractive_no_distort(xyz, camera, water_level=0, normal=(0, 0, 1), refractive_index=1.333):
    co = -camera.r.T @ camera.t

    d = co[-1] - water_level
    x = np.linalg.norm(co[:2] - xyz[:2])
    z = abs(xyz[-1] - water_level)
    u = get_u(refractive_index, d, x, z)
    oq_vec = xyz[:2] - co[:2]
    oq_vec /= np.linalg.norm(oq_vec)
    poi_xy = co[:2] + u * oq_vec
    poi = np.hstack((poi_xy, water_level, 1))
    xyw = camera.p @ poi
    xy = (xyw / xyw[-1])[:2]
    return xy



if __name__ == "__main__":
    l1 = {'unit': (1.3, 1.3, -10), 'point': (-14.2, 17, -1)}
    l2 = {'unit': (12.1, -17.2, 1.1), 'point': (1, 1, 1)}
    l3 = {'unit': (19.2, 31.8, 3.5), 'point': (2.3, 4.1, 9.8)}
    l4 = {'unit': (4, 5, 6), 'point': (1, 2, 3)}
    lines = [l1, l2, l3, l4]
    result = get_intersect_of_lines(lines).ravel()
    assert np.allclose(result, [-3.557, 7.736, 4.895], atol=0.001)
    assert np.allclose(get_trans_vec((0, 0, -3)), np.array([0, 0, -1]), atol=0.01)
