#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import fish_3d as f3
from glob import glob
import fish_track as ft
from scipy import ndimage
import matplotlib.pyplot as plt

if os.path.isfile("relinked.pkl"):
    exit()

config = ft.utility.Configure('configure.ini')

if os.path.isfile("ctraj_batches.pkl"):
    with open('ctraj_batches.pkl', 'rb') as f:
        ctraj_batches = pickle.load(f)
else:
    camera_files = glob("cam*.pkl")
    camera_files.sort()

    # collecting cameras and features in different views
    cameras, feature_handlers = [], []
    for cam_name in camera_files:
        with open(cam_name, 'rb') as f:
            cameras.append(pickle.load(f))
        cam_config = eval(f'config.{os.path.basename(cam_name)[:-4]}')
        feature_handlers.append(open(cam_config.feature, 'rb'))

    # skip initial frames
    for _ in range(config.Stereo.frame_start):
        for fh in feature_handlers:
            pickle.load(fh)

    # collection features
    features_mv_mt = []
    for frame in range(0, config.Stereo.frame_end - config.Stereo.frame_start):
        frame_mv = []
        for view in range(3):
            data = pickle.load(feature_handlers[view])
            frame = cameras[view].undistort_points(
                data[:2].T,
                want_uv=True
            )
            frame_mv.append(frame)
        features_mv_mt.append(frame_mv)

    # closing the feature handlers
    for view in range(3):
        feature_handlers[view].close()

    ctraj_batches = f3.utility.get_trajectory_batches(
        cameras,
        features_mv_mt,
        st_error_tol=config.Stereo.tol_2d,
        search_range=config.Temporal.search_range,
        tau=config.Temporal.tau,
        z_min=-config.PostProcess.water_depth,
        z_max=0,
        overlap_num=config.PostProcess.overlap_num,
        overlap_rtol=config.PostProcess.overlap_rtol,
        reproj_err_tol=config.Stereo.tol_2d
    )

    with open('ctraj_batches.pkl', 'wb') as f:
        pickle.dump(ctraj_batches, f)

if os.path.isfile("trajectories.pkl"):
    with open("trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)
else:
    ctrajs_resolved, t0_resolved = f3.utility.resolve_temporal_overlap(
        ctraj_batches,
        lag=config.Temporal.tau//2,
        ntol=config.PostProcess.overlap_num,
        rtol=config.PostProcess.overlap_rtol
    )
    trajectories = []
    for t, t0 in zip(ctrajs_resolved, t0_resolved):
        trajectories += f3.utility.convert_traj_format(t, t0)
    with open("trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)

dxs = np.arange(1, config.PostProcess.relink_dx, config.PostProcess.relink_dx_step)
for dx in dxs:
    trajectories = ft.relink_by_segments(
        trajectories,
        window_size=config.PostProcess.relink_window,
        max_frame=config.Stereo.frame_end,
        dx=dx, dt=config.PostProcess.relink_dt,
        blur_velocity=config.PostProcess.relink_blur
    )

trajectories = [t for t in trajectories if len(t[0]) > config.PostProcess.relink_min]

with open('relinked.pkl', 'wb') as f:
    pickle.dump(trajectories, f)
