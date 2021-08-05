#!/usr/bin/env python3
import os

if 'locations_3d.pkl' in os.listdir('.'):
    exit(0)

import pickle
import numpy as np
from scipy.spatial.distance import cdist
import configparser
import fish_3d as f3
from sklearn.cluster import DBSCAN


def refine(positions, cost, max_num, eps):
    """
    Refine the positions so that overlapped features will be merged.

    Args:
        positions (numpy.ndarray): a collection of nd locatiosn, shape (n, dim)
        cost (numpy.ndarray): the cost for each location, shape (n, )
            the particles with minimum cost will be returned
        max_num (int): the maximum number of refined positions
        eps (float): the distance thereshold below which
            two particles will be merged.

    Return:
        numpy.ndarray: the refined positions, shape (max_num, dim)
    """
    labels = DBSCAN(eps=eps, min_samples=1).fit(positions).labels_
    values = np.unique(labels)
    new_positions = np.empty((len(values), positions.shape[1]))  # (n, dim)
    new_cost = np.empty(values.shape)
    for i, v in enumerate(values):
        new_positions[i] = np.mean(positions[np.where(labels==v)], axis=0)
        new_cost[i] = np.mean(cost[np.where(labels==v)], axis=0)
    chosen_indices = np.argsort(new_cost)[:max_num]
    return new_positions[chosen_indices]


conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')


cameras = []
feature_handlers = []
for idx in range(1, 4):
    idx = str(idx)
    filename = conf['camera'][idx]
    if filename[-4:] == ".pkl":
        with open(filename, 'rb') as f:
            cameras.append(pickle.load(f))
    elif filename[-5:] == ".json":
        cam = f3.Camera()
        cam.load_json(filename)
        cameras.append(cam)
    else:
        exit("Can't load the camera file", filename)

    feature_handlers.append(open(conf['feature'][idx], 'rb'))


eps = int(conf['dbscan']['eps'])
max_num = int(conf['dbscan']['max_num'])

frame_start = int(conf['3d']['start'])
frame_end = int(conf['3d']['end'])
tol_2d = float(conf['3d']['tol_2d'])
overlap_3d = float(conf['3d']['overlap_3d'])
water_depth = float(conf['3d']['water_depth'])


# skip initial frames
for frame in range(frame_start):
    for i in range(len(cameras)):
        pickle.load(feature_handlers[i])


# calculating 3D location frame-by-frame 
frames = []
for f, frame in enumerate(range(frame_start, frame_end)):
    centres_multi_view = []
    for i, cam in enumerate(cameras):
        try:
            feature = pickle.load(feature_handlers[i])
        except EOFError:
            print(f"not enough featuers from view {i+1}")
            exit(1)

        centres = cam.undistort_points(feature[:2].T, want_uv=True) # shape (m, 2)
        if len(centres) > max_num:
            if (f == 0):
                cost = -feature[5]
            else:
                centres_past = centres_multi_view_history[i]
                try:
                    dist = cdist(centres, centres_past).min(axis=1)
                except:
                    dist = np.ones(feature.shape[1])
                cost = -feature[5] * dist
            centres = refine(centres, cost, max_num, eps)
        centres_multi_view.append(centres)

    centres_multi_view_history = [c for c in centres_multi_view]

    # 3D calculation
    proj_mats = [cam.p for cam in cameras]
    cam_origins = [cam.o for cam in cameras]

    if min([len(v) for v in centres_multi_view]) == 0:
        optimised = np.empty((0, 3))
        print(f'frame {frame: <10}: 0 points found')
    else:
        matched_centres, reproj_errors = f3.cstereo.locate_v3(
            *centres_multi_view, *proj_mats, *cam_origins,
            tol_2d=tol_2d, optimise=True
        )

        in_tank = matched_centres[:, 2] > -water_depth

        optimised = f3.utility.solve_overlap_lp(
            matched_centres[in_tank],
            reproj_errors[in_tank],
            overlap_3d
        )
        print(f'frame {frame: <10}: {len(matched_centres): <5} points found', end=',')
        print(f'{len(matched_centres) - np.sum(in_tank): <5} outside tank', end=',')
        print(f'{len(optimised): <5} optimised points')

    frames.append(optimised)

# clean up
for f in feature_handlers:
    f.close()

# dump 3d locations
f = open('locations_3d.pkl', 'wb')
for frame in frames:
    pickle.dump(frame, f)
f.close()
