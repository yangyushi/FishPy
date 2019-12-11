#!/usr/bin/env python3
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from typing import List
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay
from . import ray_trace
from . import camera
from typing import List, Tuple


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


def three_view_cluster_match(
    clusters_multi_view, cameras, tol_2d: float, sample_size: int, depth: float,
    report=False, normal=(0, 0, 1), water_level=0.0
    ):
    """
    match clusters tanking simutaneously by three cameras
    the clusters were assumed to be in water, n = 1.333
    """
    matched_indices, matched_centres, reproj_errors = greedy_match(
        clusters_multi_view, cameras,
        depth=depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, report=report, sample_size=sample_size,
        order=(0, 1, 2)
    )

    extra_indices_v2, extra_centres_v2, reproj_errors_v2 = greedy_match(
        clusters_multi_view, cameras,
        depth=depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, report=report, sample_size=sample_size,
        order=(1, 0, 2), history=matched_indices
    )

    if len(extra_indices_v2) > 0:
        matched_indices = np.concatenate((matched_indices, extra_indices_v2))
        matched_centres = np.concatenate((matched_centres, extra_centres_v2))
        reproj_errors = np.concatenate((reproj_errors, reproj_errors_v2))

    extra_indices_v3, extra_centres_v3, reproj_errors_v3 = greedy_match(
        clusters_multi_view, cameras,
        depth=depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, report=report, sample_size=sample_size,
        order=(2, 0, 1), history=matched_indices
    )

    if len(extra_indices_v3) > 0:
        matched_indices = np.concatenate((matched_indices, extra_indices_v3))
        matched_centres = np.concatenate((matched_centres, extra_centres_v3))
        reproj_errors = np.concatenate((reproj_errors, reproj_errors_v3))

    return matched_indices, matched_centres, reproj_errors


def greedy_match(clusters, cameras, depth, normal, water_level, tol_2d, sample_size=10, report=True, order=(0, 1, 2), history=np.empty((0, 3))):
    """
    use greedy algorithm to match clusters across THREE views

    start from view_1:
        start from cluster_1:
            1. find all possible correspondance according to epipolar relationship in view 2 and view 3
            2. for all possibility in view 2, validate according to ray-tracing
            3. for all possibility in view 3, validate according to ray-tracing

    :param clusters: a collection of points in the 2D image with the format of (u, v), NOT (x, y)
    :param cameras: a collection of Camera instances
    :param depth: the maximum depth of water used to constrain the length of the epipolar relation
    :param normal: the direction of the normal of the water. It should be [0, 0, 1]
    :param tol_2d: tolerance on the distance between epipolar line and pixels
    :param points: the number of points used in 3D stereo matching for each cluster
    :return: the matched indices across different views. The indices are for the clusters.
    """
    matched = []
    centres = []
    reproj_errors = []
    cameras_reordered = [cameras[i] for i in order]
    clusters = [clusters[i] for i in order]

    centres_mv = [[
            cluster.mean(0) for cluster in cluster_one_view
        ] for cluster_one_view in clusters]

    for i, centre in enumerate(centres_mv[0]):

        a12, b12 = ray_trace.epipolar_la(centre, cameras_reordered[0], cameras_reordered[1], water_level, depth, normal)
        a13, b13 = ray_trace.epipolar_la(centre, cameras_reordered[0], cameras_reordered[2], water_level, depth, normal)

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

        # remove points in view 2 that can't match anything in view 3
        to_delete = np.ones(len(candidates_12), dtype=bool)
        for j, c2 in enumerate(candidates_12):
            centre_2 = centres_mv[1][c2]
            a23, b23 = ray_trace.epipolar_la(
                    centre_2, cameras_reordered[1], cameras_reordered[2], water_level, depth, normal
                    )
            for c3 in candidates_13:
                cluster = clusters[2][c3]
                distances = np.abs(cluster.T[0] * a23 - cluster.T[1] + b23) / np.sqrt(a23**2 + 1)
                if np.min(distances) < tol_2d:
                    to_delete[j] = False
                    break
        if to_delete.all():
            if report:
                print("no 3D clouds")
            continue
        candidates_12 = np.array(candidates_12)[~to_delete]

        # remove points in view 3 that can't match anything in view 2
        to_delete = np.ones(len(candidates_13), dtype=bool)
        for j, c3 in enumerate(candidates_13):
            centre_3 = centres_mv[2][c3]
            a32, b32 = ray_trace.epipolar_la(
                    centre_3, cameras_reordered[2], cameras_reordered[1], water_level, depth, normal
                    )
            for c2 in candidates_12:
                cluster = clusters[1][c2]
                distances = np.abs(cluster.T[0] * a32 - cluster.T[1] + b32) / np.sqrt(a32**2 + 1)
                if np.min(distances) < tol_2d:
                    to_delete[j] = False
                    break
        if to_delete.all():
            if report:
                print("no 3D clouds")
            continue
        candidates_13 = np.array(candidates_13)[~to_delete]

        # get xyz coordinates from 3 views
        candidates = list(itertools.product([i], candidates_12, candidates_13))
        allowed, coms, errors = [], [], []

        for candidate in candidates:
            test = history.T - np.vstack(candidate)
            if (test == 0).any():
                continue
            full_clusters = [
                    clusters[0][candidate[0]],
                    clusters[1][candidate[1]],
                    clusters[2][candidate[2]]
            ]
            par_clusters = list(map(lambda x: get_partial_cluster(x, sample_size), full_clusters))
            cloud, error = match_clusters(par_clusters, cameras_reordered, normal, water_level)

            z = np.sum(cloud.T[-1] / error) / np.sum(1/error)  # weighted by inverse error
            in_tank = (z < water_level) and (z > -depth)
            in_tank = True

            if len(cloud) > 0 and in_tank:
                cloud_com = np.sum(cloud / np.reshape(error, (len(error), 1)), axis=0) / np.sum(1 / error)
                locations_2d = [np.mean(full_clusters[i], 0) for i in range(3)]
                try:
                    reproj_err = ray_trace.get_reproj_err(cloud_com, locations_2d, cameras_reordered, water_level, normal)
                    candidate_origional = [candidate[order.index(i)] for i in range(3)]
                    allowed.append(candidate_origional)
                    errors.append(reproj_err)
                    coms.append(cloud_com)
                except ValueError:
                    continue

        if len(allowed) > 0:
            matched.append(tuple(allowed[np.argmin(errors)]))
            reproj_errors.append(np.min(errors))
            centres.append(coms[np.argmin(errors)])
        if report:
            print(f"{len(matched)} 3D clouds found")
    if len(centres) > 0:
        centres = np.array(centres)
        matched = np.array(matched)
    else:
        centres = np.zeros((0, 3))
        matched = np.zeros((0, 3), dtype=int)
    return matched, centres, np.array(reproj_errors)


def match_clusters(clusters, cameras, normal, water_level):
    """
    return allowed 3d points given matched clusters in different views
    the error is the average of perpendicular of 3D points to the three rays, unit is mm
    """
    xyz, err = ray_trace.ray_trace_refractive_cluster(
            clusters, cameras, z=water_level, normal=normal
            )
    return xyz, err


def remove_overlap(centres, errors, search_range=10):
    """
    centres: possible fish locations, array shape (number, dimension)
    errors: reprojection errors of COM of different cloudsshape, shape (number, )
    overlap = convex hull overlap; vertices of ch1 goes into ch2
    """
    overlapped = []
    for i, c1 in enumerate(centres):
        for j, c2 in enumerate(centres[i+1:]):
            if np.linalg.norm(centres[i] - centres[i+j+1]) <= search_range:
                overlapped.append((i, j+i+1))
    if len(overlapped) > 0:
        overlapped = join_pairs(overlapped)
    else:
        return centres  # do not remove anything
    not_overlapped = [i for i in range(len(centres)) if not (i in np.hstack(overlapped))]
    for group in overlapped:
        best = np.argmin(errors[np.array(group)])
        not_overlapped.append(group[best])
    return centres[np.array(not_overlapped)]


def remove_conflict(matched_indices, matched_centres, reproj_errors):
    """
    Only allow each unique feature, indicated by a number in `matched_indices`, appear once for each view
    This means all values in matched_indices.T should be unique along three views
    It works for n views
    matched_indices: shape (number, view)
    """
    mask = np.ones(len(matched_centres), dtype=bool)
    view_num = matched_indices.shape[1]

    for view in range(view_num):
        unique_vals, counts = np.unique(matched_indices[:, view], return_counts=True)
        duplicate = unique_vals[counts > 1]

        for val in duplicate:
            indices = np.where(matched_indices[:, view] == val)[0]
            best = np.argmin(reproj_errors[indices])
            for j, i in enumerate(indices):
                if j != best:
                    mask[i] = 0

    matched_indices = matched_indices[mask>0]
    matched_centres = matched_centres[mask>0]
    reproj_errors = reproj_errors[mask>0]
    return matched_indices, matched_centres, reproj_errors


def extra_three_view_cluster_match(
    matched_indices, clusters_multi_view, cameras, tol_2d: float, sample_size: int, depth: float,
    report=False, normal=(0, 0, 1), water_level=0.0
    ):
    """
    match clusters tanking simutaneously by three cameras
    pretending only features that DID NOT correspond to indices in  `matched_indices` appear in the view
    matched_indices: shape (number, view)
    """
    clusters_left = []
    indices_left_multi_view = []
    for view, indices_matched in enumerate(matched_indices.T):
        clusters_left.append([])
        cluster_num = len(clusters_multi_view[view])
        indices_full = np.arange(cluster_num)
        indices_left = np.setdiff1d(indices_full, indices_matched)
        for i in indices_left:
            clusters_left[-1].append(clusters_multi_view[view][i])
        indices_left_multi_view.append(indices_left)

    matched_indices, matched_centres, reproj_errors = greedy_match(
        clusters_left, cameras,
        depth=depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, report=report, sample_size=sample_size,
        order=(0, 1, 2)
    )

    extra_indices_v2, extra_centres_v2, reproj_errors_v2 = greedy_match(
        clusters_left, cameras,
        depth=depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, report=report, sample_size=sample_size,
        order=(1, 0, 2), history=matched_indices
    )

    if len(extra_indices_v2) > 0:
        matched_indices = np.concatenate((matched_indices, extra_indices_v2))
        matched_centres = np.concatenate((matched_centres, extra_centres_v2))
        reproj_errors = np.concatenate((reproj_errors, reproj_errors_v2))

    extra_indices_v3, extra_centres_v3, reproj_errors_v3 = greedy_match(
        clusters_left, cameras,
        depth=depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, report=report, sample_size=sample_size,
        order=(2, 0, 1), history=matched_indices
    )

    if len(extra_indices_v3) > 0:
        matched_indices = np.concatenate((matched_indices, extra_indices_v3))
        matched_centres = np.concatenate((matched_centres, extra_centres_v3))
        reproj_errors = np.concatenate((reproj_errors, reproj_errors_v3))

    for view in range(matched_indices.shape[1]):
        matched_indices[:, view] = indices_left_multi_view[view][matched_indices[:, view]]

    return matched_indices, matched_centres, reproj_errors


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


def match_points_v3(cameras: List['Camera'], points: List[np.ndarray], max_cost=1):
    """
    match points in three views
    no refractions were considered
    :param cameras: a list of three cameras
    :param points: (x, y) coordinates in three views
    """
    # (n, 3) & (n, 1)
    points_homo = [np.concatenate(
        (p, np.ones((len(p), 1))), axis=1
    ) for p in points]
    results = []
    f12 = camera.get_fundamental( cameras[0], cameras[1] )
    f13 = camera.get_fundamental( cameras[0], cameras[2] )
    f23 = camera.get_fundamental( cameras[1], cameras[2] )
    f12 = f12 / f12.max()
    f13 = f13 / f13.max()
    f23 = f23 / f23.max()
    results = []
    for p1h in points_homo[0]:
        cost_v2 = np.array([p2h @ f12 @ p1h for p2h in points_homo[1]])
        cost_v3 = np.array([p3h @ f13 @ p1h for p3h in points_homo[2]])
        mc2 = np.min(np.abs(cost_v2))
        mc3 = np.min(np.abs(cost_v3))
        
        best_v2 = np.argmin(np.abs(cost_v2))
        best_v3 = np.argmin(np.abs(cost_v3))
        
        mc3 = np.min(np.abs(cost_v3))
        
        p2h = points_homo[1][best_v2]
        p3h = points_homo[2][best_v3]
        mc23 = abs(p3h @ f23 @ p2h)

        if max(mc2, mc3, mc23) < max_cost:
            p3d = triangulation_v3(
                (p1h, p2h, p3h),
                cameras
            )
            results.append(p3d)
    return np.array(results)
