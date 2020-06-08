import pickle
import numpy as np
import fish_3d as f3
import fish_track as ft
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np


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



cameras = []
for cam_name in [f'cam_{i}.pkl' for i in range(1, 4)]:
    with open(cam_name, 'rb') as f:
        cameras.append(pickle.load(f))

centres_1 = open('features_2d-cam_1.pkl', 'rb')
centres_2 = open('features_2d-cam_2.pkl', 'rb')
centres_3 = open('features_2d-cam_3.pkl', 'rb')

for _ in range(10000):
    pickle.load(centres_1)
    pickle.load(centres_2)
    pickle.load(centres_3)

frames = 150

features_mv_mt = []
for frame in range(0, frames):
    features_mv_mt.append([
        cameras[0].undistort_points(pickle.load(centres_1)[:2].T, want_uv=True),
        cameras[1].undistort_points(pickle.load(centres_2)[:2].T, want_uv=True),
        cameras[2].undistort_points(pickle.load(centres_3)[:2].T, want_uv=True),
    ])

centres_1.close()
centres_2.close()
centres_3.close()

trajectories = get_short_trajs(
    cameras, features_mv_mt, st_error_tol=5, search_range=20, t1=5, t2=3,
    z_min=-350, z_max=0, overlap_num=5, overlap_rtol=10
)

relinked = ft.relink(trajectories, dx=10, dt=5)


with open('trajectories.pkl', 'wb') as f:
    pickle.dump(relinked, f)
