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


def get_poi(p, z, coordinate):
    """
    - p: the projection matrix of the camera
    - z: the hight of water level with respect to world origin
    - corrdinate: the position (x, y) of object on the image, unit is pixel
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

    The equation: P @ [x, y, z, 1]' = [c * v, c * u, c]' are solved for x, y, c, knowing everything else
    """
    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = p.ravel()
    u, v = coordinate

    x =  (z*p12*p23 - z*p12*p33*v - z*p13*p22 + z*p13*p32*v + z*p22*p33*u - \
          z*p23*p32*u + p12*p24 - p12*p34*v - p14*p22 + p14*p32*v + p22*p34*u - p24*p32*u) /\
         (p11*p22 - p11*p32*v - p12*p21 + p12*p31*v + p21*p32*u - p22*p31*u)

    y = -(z*p11*p23 - z*p11*p33*v - z*p13*p21 + z*p13*p31*v + z*p21*p33*u - \
          z*p23*p31*u + p11*p24 - p11*p34*v - p14*p21 + p14*p31*v + p21*p34*u - p24*p31*u) /\
         (p11*p22 - p11*p32*v - p12*p21 + p12*p31*v + p21*p32*u - p22*p31*u)
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
    return t / np.linalg.norm(t)


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
    poi_1 = get_poi(camera_1.p, interface, uv[::-1])
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
    return all the pixels that contains the epipolar line in image_2

    The meanings of variable names can be found in this paper

    :param uv: (u, v) location of a pixel on image from camera_1, NOT (x, y)
    :param interface: height of water level in WORLD coordinate system
    :param normal: normal of the air-water interface, from water to air
    :param n: refractive index of water (or some other media)
    
    """
    
    co_1 = -camera_1.r.T @ camera_1.t  # camera origin
    co_2 = -camera_2.r.T @ camera_2.t  # camera origin
    poi_1 = get_poi(camera_1.p, interface, uv[::-1])
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
        u = optimize.root_scalar(find_u, args=(n, d, x, z), x0=x/1.5, x1=x).root
        if (u > x) or (u < 0):
            print("root finding in refractive epipolar geometry fails")
            break
        o = np.hstack((co_2[:2], interface))
        oq_vec = np.hstack((m[:2], interface)) - o
        q = o + u * (oq_vec / np.linalg.norm(oq_vec))
        xy_2 = camera_2.p @ np.hstack((q, 1))
        xy_2 /= xy_2[-1]
        xy_2 = np.floor(xy_2[:2]).astype(int)
        uv_2 = xy_2[::-1]

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


def ray_trace_refractive(centres, cameras, z=0, normal=(0, 0, 1), refractive_index=1.33):
    """
    :param centres: a list of centres (u, v) in different views
    :param cameras: a list of *calibrated* Camera objects
    :param z: the z-value of the refractive interface
    """
    camera_origins = [-camera.r.T @ camera.t for camera in cameras]
    pois = [
        get_poi(camera.p, z=z, coordinate=centre[::-1]) for camera, centre in zip(cameras, centres)
    ]  # ::-1 changes from (u, v) to (x, y)
    incid_rays = [poi - co for poi, co in zip(pois, camera_origins)]
    trans_rays = [get_trans_vec(incid, normal=normal) for incid in incid_rays]
    trans_lines = [{'unit': t, 'point': poi} for t, poi in zip(trans_rays, pois)]
    
    point_3d = get_intersect_of_lines(trans_lines)
    
    error = 0
    for line in trans_lines:
        error += pl_dist(point_3d, line)
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
    poi = np.hstack((poi_xy, water_level, 1))
    img_xy = camera.p @ poi
    img_xy = (img_xy / img_xy[-1])[:2]
    return img_xy

if __name__ == "__main__":
    l1 = {'unit': (1.3, 1.3, -10), 'point': (-14.2, 17, -1)}
    l2 = {'unit': (12.1, -17.2, 1.1), 'point': (1, 1, 1)}
    l3 = {'unit': (19.2, 31.8, 3.5), 'point': (2.3, 4.1, 9.8)}
    l4 = {'unit': (4, 5, 6), 'point': (1, 2, 3)}
    lines = [l1, l2, l3, l4]
    result = get_intersect_of_lines(lines).ravel()
    assert np.allclose(result, [-3.557, 7.736, 4.895], atol=0.001)
    assert np.allclose(get_trans_vec((0, 0, -3)), np.array([0, 0, -1]), atol=0.01)
