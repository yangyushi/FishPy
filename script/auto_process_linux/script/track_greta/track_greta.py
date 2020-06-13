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

if os.path.isfile("trajectories.pkl"):
    with open('trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)
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
            frame = cameras[view].undistort_points(
                pickle.load(feature_handlers[view])[:2].T,
                want_uv=True
            )
            frame_mv.append(frame)
        features_mv_mt.append(frame_mv)

    # closing the feature handlers
    for view in range(3):
        feature_handlers[view].close()

    trajectories = f3.utility.get_short_trajs(
        cameras, features_mv_mt,
        st_error_tol=config.Stereo.tol_2d,
        search_range=config.Temporal.search_range,
        t1=config.Temporal.t1,
        t2=config.Temporal.t2,
        z_min=-config.PostProcess.water_depth,
        z_max=0,
        overlap_num=config.PostProcess.overlap_num,
        overlap_rtol=config.PostProcess.overlap_rtol
    )

    with open('trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

relinked = ft.relink(trajectories, 1, 1, config.PostProcess.relink_blur)
dxs = np.arange(1, config.PostProcess.relink_dx, config.PostProcess.relink_dx_step)
for dx in dxs:
    relinked = ft.relink( relinked, dx, config.PostProcess.relink_dt, None)

with open('relinked.pkl', 'wb') as f:
    pickle.dump(relinked, f)
