#!/usr/bin/env python3
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from . import ray_trace
from typing import List


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


def get_partial_cluster(cluster, size):
    l = len(cluster)
    if l <= size:
        return cluster
    elif l % size == 0:
        return cluster[::(l // size)]
    else:
        return cluster[:-(l % size):(l // size)]


def greedy_match_centre(clusters, cameras, images, depth, normal, water_level, tol_2d, tol_3d, points=10, report=True):
    """
    use greedy algorithm to match clusters across THREE views

    start from view_1:
        start from cluster_1:
            1. find all possible correspondance according to epipolar relationship in view 2 and view 3
            2. for all possibility in view 2, validate according to ray-tracing
            3. for all possibility in view 3, validate according to ray-tracing

    :param clusters: a collection of points in the 2D image with the format of (u, v), NOT (x, y)
    :param cameras: a collection of Camera instances
    :param images: multiple images in views of different cameras
    :param depth: the maximum depth of water used to constrain the length of the epipolar relation
    :param normal: the direction of the normal of the water. It should be [0, 0, 1]
    :param tol_2d: tolerance on the distance between epipolar line and pixels
    :param tol_3d: tolerance on the distance between 3D rays
    :param points: the number of points used in 3D stereo matching for each cluster
    :return: the matched indices across different views. The indices are for the clusters.
    """
    matched = []

    centres_mv = [[
            cluster.mean(0) for cluster in cluster_one_view
        ] for cluster_one_view in clusters]  # (v, u)

    for i, centre in enumerate(centres_mv[0]):

        a12, b12 = ray_trace.epipolar_la(centre, cameras[0], cameras[1], images[1], water_level, depth, normal)
        a13, b13 = ray_trace.epipolar_la(centre, cameras[0], cameras[2], images[2], water_level, depth, normal)

        candidates_12, candidates_13 = [], []

        for j, cluster in enumerate(clusters[1]):
            distances = np.abs(cluster.T[0] * a12 - cluster.T[1] + b12) / np.sqrt(a12**2 + 1)
            if np.min(distances) < tol_2d:
                candidates_12.append(j)

        for j, cluster in enumerate(clusters[2]):
            distances = np.abs(cluster.T[0] * a13 - cluster.T[1] + b13) / np.sqrt(a13**2 + 1)
            if np.min(distances) < tol_2d:
                candidates_13.append(j)

        if report:
            print(f'#{i}, candidates in camera #2 is {len(candidates_12)}, candidates in camera #3 is {len(candidates_13)}',
                    end=' --> ')
        if not (candidates_12 and candidates_13):
            if report:
                print("no 3D clouds")
            continue

        candidates = list(itertools.product([i], candidates_12, candidates_13))
        clouds_3d = []
        for candidate in candidates:
            full_clusters = [
                    clusters[0][candidate[0]],
                    clusters[1][candidate[1]],
                    clusters[2][candidate[2]]
            ]
            #min_num = np.min([len(f) for f in full_clusters])
            #min_num = np.min([min_num, points])
            par_clusters = map(lambda x: get_partial_cluster(x, points), full_clusters)
            cloud = match_clusters_batch(par_clusters, cameras, normal, water_level, tol_3d)
            if len(cloud) > 0:
                matched.append(candidate)
        if report:
            print(f"{len(matched)} 3D clouds found")
    return matched


def match_clusters_batch(clusters, cameras, normal, water_level, tol):
    results = []
    xyz, err = ray_trace.ray_trace_refractive_cluster(
            clusters, cameras, z=water_level, normal=normal
            )
    return xyz[err < tol]


def match_clusters(clusters, cameras, normal, water_level, tol):
    combinations = list(itertools.product(*clusters))
    results = []
    # todo: write a batch version for this
    for comb in combinations:
        xyz, err = ray_trace.ray_trace_refractive_faster(comb, cameras, z=water_level, normal=normal)
        if err < tol:
            results.append(xyz)
    return np.array(results)


def match_clusters_faster(clusters, cameras, normal, water_level, tol):
    xyz, err = ray_trace.ray_trace_refractive_faster(clusters, cameras, z=water_level, normal=normal)
    xyz = xyz[np.where(err < tol)]
    return np.array(results)


def reconstruct_clouds(cameras, matched_indices, clusters_multi_view,
        water_level, normal, sample_size, tol):
    clouds = []
    for indices in matched_indices:
        i1, i2, i3 = indices
        full_clusters = (
            clusters_multi_view[0][i1],
            clusters_multi_view[1][i2],
            clusters_multi_view[2][i3])

        par_clusters = map(lambda x: get_partial_cluster(x, sample_size), full_clusters)
        xyz, err = ray_trace.ray_trace_refractive_cluster(
                par_clusters, cameras, z=water_level, normal=normal
        )

        cloud = xyz[err < tol]
        if len(cloud) > 0:
            clouds.append(np.array(cloud))
    return clouds


def merge_clouds(clouds, min_dist, min_num):
    """
    close pair = two points whose distance is smaller than min_dist
    overlapped clouds = two clouds that have more than *min_num* close paris
    merge all overlapped clouds
    """
    overlapped = []
    for i, c1 in enumerate(clouds):
        for j, c2 in enumerate(clouds[i+1:]):
            dist = cdist(c1, c2)
            indices = np.triu_indices_from(dist, k=1)
            dist = dist[indices]
            if np.sum(dist < min_dist) > min_num:
                overlapped.append((i, j+i+1))
    if len(overlapped) > 0:
        overlapped = join_pairs(overlapped)

        to_del = []
        for labels in overlapped:
            new_p1 = np.concatenate([clouds[l] for l in labels], axis=0)
            clouds[labels[0]] = new_p1
            for l in labels[1:]:
                to_del.append(l)
        clouds = np.delete(clouds, to_del)
    return clouds


def triangulation_v3(positions: np.ndarray, cameras: List['Camera']):
    """
    the positions in different views should be undistorted
    """
    M = np.ones((9, 7), dtype=np.float64)
    M[:3, :4] = cameras[0].p
    M[3:6, :4] = cameras[1].p
    M[6:9, :4] = cameras[2].p
    M[:3, 4] = - positions[0]
    M[3:6, 5] = - positions[1]
    M[6:9, 6] = - positions[2]
    U, S, V = np.linalg.svd(M)
    X = V[-1, :4]
    X = X / X[-1]
    return X[:3]


def join_pairs(pairs):
    if len(pairs) == 0:
        return []
    max_val = np.max(np.hstack(pairs)) + 1
    canvas = np.zeros((max_val, max_val), dtype=int)
    p = np.array(pairs)
    canvas[tuple(p.T)] = 1
    labels, _ = ndimage.label(canvas)
    joined_pairs = []
    for val in set(labels[labels > 0]):
        joined_pair = np.unique(np.vstack(np.where(labels == val)))
        joined_pairs.append(joined_pair)
    return joined_pairs
