#!/usr/bin/env python3
import numpy as np
from numba import njit
from scipy import optimize
from itertools import product
import cv2


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


def get_intersect_of_lines_batch(lines):
    """
    lines = [line, ...], shape -> (n, view, 2, 3)
    line = [points (a), unit directions (v)]
    M(3, 3) @ x(3, n) = b(3, n)
    """
    M = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,  ), dtype=np.float64)
    xyz = []
    view_num = len(lines[0])
    for line in lines:
        M *= 0
        b *= 0
        d = 0
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
    dists = []
    for v in range(view_num):
        delta = points - lines[:, v, 0, :]  # (num, 0)
        d = np.linalg.norm(np.cross(delta, lines[:, v, 1, :]), axis=-1)
        dists.append(d)
    dists = np.mean(dists, 0)  # (view, n, dim) -> (n, dim)
    return dists


def get_poi(camera, z, coordinate):
    """
    - camera: a Camera instance
    - z: the hight of water level with respect to world origin
    - corrdinate: the position (u, v) of object on the image, unit is pixel, NOT (x, y)
        it can be one point or many points
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
    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = camera.ext.ravel()

    if coordinate.ndim == 1:  # (just one point)
        x, y = camera.undistort(coordinate[::-1])  # return (v, u) -> (x, y)
        z = z
    elif coordinate.ndim == 2:  # (number, dim)
        tmp = np.flip(coordinate.copy(), 1)  # (v, u) -> (u, v)
        x, y = camera.undistort_points(tmp)  # return (u, v) -> (x, y)
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
    u0, u1 = 0, image.shape[0]
    v0, v1 = 0, image.shape[1]
    u, v = uv
    is_inside = True
    is_inside *= u > u0
    is_inside *= u < u1
    is_inside *= v > v0
    is_inside *= v < v1
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


def epipolar_draw(vu, camera_1, camera_2, image_2, interface=0, depth=400, normal=(0, 0, 1), n=1.33):
    """
    return all the pixels that contains the epipolar line in image_2, re-projected & distorted

    The meanings of variable names can be found in this paper

    :param vu: (v, u) location of a pixel on image from camera_1, NOT (u, v)
    :param interface: height of water level in WORLD coordinate system
    :param normal: normal of the air-water interface, from water to air
    :param n: refractive index of water (or some other media)
    """
    poi_1 = get_poi(camera_1, interface, np.array(vu))
    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    incid = poi_1 - co_1
    trans = get_trans_vec(incid, normal=normal)
    is_in_image = True
    m = poi_1.copy()
    epipolar_pixels = []
    step = 1
    count = 0
    last_vu = None

    while is_in_image:
        m += step * trans
        z = abs(m[-1])
        if z > depth:
            break
        d = abs(co_2[-1] - interface)
        x = np.linalg.norm(m[:2] - co_2[:2])
        u = optimize.root_scalar(
                find_u, args=(n, d, x, z), x0=x*0.8, x1=x*0.5
                ).root
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
        vu_2 = uv_2[::-1]

        is_in_image = is_inside_image(vu_2, image_2)

        if count == 0:
            is_connected = True
            is_different = True
        else:
            is_connected = np.sum(np.power(vu_2 - last_vu, 2)) <= 2
            is_different = not np.allclose(vu_2, last_vu)

        should_draw = is_different and is_in_image and is_connected

        if should_draw:
            epipolar_pixels.append(vu_2.copy())
            epipolar_pixels.append(vu_2.copy() + 1)
            last_vu = vu_2.copy()
        else:
            if not is_different:
                step *= 2
            if not is_connected:
                m -= step * trans
                step /= 3
        count += 1

    return np.array(epipolar_pixels)


def epipolar_la(vu, camera_1, camera_2, image_2, interface=0, depth=400, normal=(0, 0, 1), n=1.33):
    """
    linear approximation for epipolar line under water
    use 3 epipolar points under water and do a linear fit
    """
    poi_1 = get_poi(camera_1, interface, np.array(vu))
    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    incid = poi_1 - co_1
    trans = get_trans_vec(incid, normal=normal)
    X = np.zeros((3, 2))
    Y = np.zeros((3, 1))
    for i, step in enumerate([0, depth/2, depth]):
        m = poi_1 + step * trans
        z = abs(m[-1])
        d = abs(co_2[-1] - interface)
        x = np.linalg.norm(m[:2] - co_2[:2])
        u = optimize.root_scalar(
                find_u, args=(n, d, x, z), x0=x*0.8, x1=x*0.5
                ).root
        if (u > x) or (u < 0):
            print("root finding in refractive epipolar geometry fails")
            break
        o = np.hstack((co_2[:2], interface))
        oq_vec = np.hstack((m[:2], interface)) - o
        q = o + u * (oq_vec / np.linalg.norm(oq_vec))
        uv_2 = camera_2.project(q)
        X[i, :] = uv_2[1], 1  # uv -> vu
        Y[i, 0] = uv_2[0]
    a, b = (np.linalg.inv(X.T @ X) @ X.T) @ Y  # least square fit
    return a, b



def ray_trace_refractive_cluster(clusters, cameras, z=0, normal=(0, 0, 1), refractive_index=1.33):
    camera_origins = [np.vstack(-camera.r.T @ camera.t) for camera in cameras]  # 3 coordinates whose shape is (3, 1)
    pois_mv = []
    for camera, cluster in zip(cameras, clusters):
        pois = get_poi(camera, z=z, coordinate=cluster).T
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


def ray_trace_refractive(centres, cameras, z=0, normal=(0, 0, 1), refractive_index=1.33):
    """
    :param centres: a list of centres (v, u) in different views, they should NOT be undistorted
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
    :param centres: a list of centres (v, u) in different views, they should NOT be undistorted
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
    camera_origin = -camera.r.T @ camera.t

    c_ = (
            { 'type': 'ineq', 'fun': lambda v1, v2, v3: same_direction(v1, v2, v3, 0), 'args': (xyz, camera_origin) },
            { 'type': 'ineq', 'fun': lambda v1, v2, v3: same_direction(v1, v2, v3, 1), 'args': (xyz, camera_origin) },
            )

    result = optimize.minimize(
            cost_snell,
            x0=(xyz[0], xyz[1]),
            args=(water_level, xyz, camera_origin, np.array(normal), refractive_index),
            method='SLSQP',
            constraints=c_
            )
    poi_xy = result.x
    poi = np.hstack((poi_xy, water_level))
    uv_2, _ = cv2.projectPoints(
            objectPoints=np.vstack(poi).T,
            rvec=camera.rotation.as_rotvec(),
            tvec=camera.t,
            cameraMatrix=camera.k,
            distCoeffs=camera.distortion
    )
    return uv_2.ravel()

if __name__ == "__main__":
    l1 = {'unit': (1.3, 1.3, -10), 'point': (-14.2, 17, -1)}
    l2 = {'unit': (12.1, -17.2, 1.1), 'point': (1, 1, 1)}
    l3 = {'unit': (19.2, 31.8, 3.5), 'point': (2.3, 4.1, 9.8)}
    l4 = {'unit': (4, 5, 6), 'point': (1, 2, 3)}
    lines = [l1, l2, l3, l4]
    result = get_intersect_of_lines(lines).ravel()
    assert np.allclose(result, [-3.557, 7.736, 4.895], atol=0.001)
    assert np.allclose(get_trans_vec((0, 0, -3)), np.array([0, 0, -1]), atol=0.01)
