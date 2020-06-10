import os
import pickle
import numpy as np
import fish_3d as f3
from glob import glob
import fish_track as ft
from scipy import ndimage
import matplotlib.pyplot as plt


def get_short_trajs(
    cameras, features_mv_mt, st_error_tol, search_range, t1, t2,
    z_min, z_max, overlap_num, overlap_rtol ):
    shift = t1 * t2
    frame_num = len(features_mv_mt)
    t_starts = [t * shift for t in range(frame_num // shift)]
    proj_mats = [cam.p for cam in cameras]
    cam_origins = [cam.o for cam in cameras]
    trajectories = []
    for t0 in t_starts:
        print("processing ", t0, " - ", t0 + shift)
        stereo_matches = []
        for features_mv in features_mv_mt[t0 : t0 + shift]:
            matched = f3.cstereo.match_v3(
                *features_mv, *proj_mats, *cam_origins,
                tol_2d=st_error_tol, optimise=True
            )
            stereo_matches.append(matched)
        features_mt_mv = []  # shape (3, frames, n, 3)
        for view in range(3):
            features_mt_mv.append([])
            for frame in range(t0, t0 + shift):
                features_mt_mv[-1].append( features_mv_mt[frame][view] )
        try:
            ctrajs_3d = f3.cgreta.get_trajs_3d_t1t2(
                features_mt_mv, stereo_matches, proj_mats, cam_origins, c_max=500,
                search_range=search_range, search_range_traj=search_range,
                tau_1=t1, tau_2=t2
            )

            trajs_3d_opt = f3.utility.post_process_ctraj(
                ctrajs_3d, t0, z_min, z_max,
                overlap_num, overlap_rtol
            )
            trajectories += trajs_3d_opt
        except:
            print(f"Tracking error from {t0} - {t0 + shift}")
    return trajectories


if __name__ == "__main__":

    config = ft.utility.Configure('configure.ini')

    camera_files = glob("cam*.pkl")
    cameras_files.sort()

    # collecting cameras and features in different views
    cameras, features_handlers = [], []
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
                pickle.load(features_handlers[view])[:2].T,
                want_uv=True
            )
            frame_mv.append(frame)
        frames_mv_mt.append(frame_mv)

    # closing the feature handlers
    for view in range(3):
        features_handlers[view].close()

    trajectories = get_short_trajs(
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

    relinked = ft.relink(
        trajectories,
        dx=config.PostProcess.relink_dx,
        dt=config.PostProcess.relink_dt
    )

    with open('trajectories.pkl', 'wb') as f:
        pickle.dump(relinked, f)
