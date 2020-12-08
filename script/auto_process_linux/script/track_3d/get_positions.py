#!/usr/bin/env python3
import os
import glob
import pickle
import numpy as np
import configparser
import fish_3d as f3
import fish_track as ft
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


config = ft.utility.Configure('configure.ini')

if 'cameras.pkl' in os.listdir('.'):
    with open(f'cameras.pkl', 'rb') as f:
        camera_dict = pickle.load(f)
elif 'cameras.pkl' in os.listdir('result'):
    with open(f'result/cameras.pkl', 'rb') as f:
        camera_dict = pickle.load(f)
else:
    raise FileNotFoundError("Can't locate calibrated cameras, run make calib")

frame_start = config.Stereo.frame_start
frame_end = config.Stereo.frame_end
normal = (0, 0, config.Stereo.normal)
water_level = config.Stereo.water_level
water_depth = config.Stereo.water_depth
sample_size = config.Stereo.sample_size
tol_2d = config.Stereo.tol_2d
see_reprojection = bool(config.Plot.see_reprojection)

cameras = []
camera_configurations = []
feature_handlers = []
if see_reprojection:
    movies = []
for i, cam_name in enumerate(camera_dict):
    cam_config = eval(f'config.{cam_name}')
    cam = camera_dict[cam_name]
    cameras.append(cam)
    camera_configurations.append(cam_config)
    feature_handlers.append(open(cam_config.feature, 'rb'))
    if see_reprojection:
        movies.append( ft.read.iter_video(cam_config.images) )

for frame in range(frame_start):  # skip initial frames
    for i in range(len(cameras)):
        if see_reprojection:
            next(movies[i])
        pickle.load(feature_handlers[i])

frames = []
for frame in range(frame_start, frame_end):
    if see_reprojection:
        images_multi_view = []
    features_multi_view = []
    clusters_multi_view = []
    centres_multi_view = []

    for i, cam in enumerate(cameras):
        cam_config = camera_configurations[i]

        # the degree 180 is not included, it should be covered by another "upside-down" shape
        angle_number = cam_config.orientation_number
        angles = np.linspace(0, 180, angle_number)

        shape_kernels = np.load(cam_config.shape)

        if see_reprojection:
            image = next(movies[i])

        try:
            feature = pickle.load(feature_handlers[i])
        except EOFError:
            print(f"not enough featuers from view {i+1}")
            break

        clusters = ft.oishi.get_clusters(
            feature, shape_kernels, angles,
            kernel_threshold=config.Stereo.kernel_threshold
        )
        clusters = [cam.undistort_points(c, want_uv=True) for c in clusters]  # undistort each cluster
        centres = np.array([c.mean(axis=0) for c in clusters])   # shape (m, 2)

        if see_reprojection:
            images_multi_view.append(image)
        clusters_multi_view.append(clusters)
        features_multi_view.append(feature)
        centres_multi_view.append(centres)

    proj_mats = [cam.p for cam in cameras]
    cam_origins = [cam.o for cam in cameras]

    matched_centres, reproj_errors = f3.cstereo.locate_v3(
        *centres_multi_view, *proj_mats, *cam_origins,
        tol_2d=tol_2d, optimise=True
    )

    in_tank = matched_centres[:, 2] > -water_depth

    print(f'frame {frame: <10}: {len(matched_centres): <5} points found, {len(matched_centres) - np.sum(in_tank): <5} outside tank')

    frames.append(matched_centres[in_tank])

    if see_reprojection:
        f3.utility.plot_reproject(
            images_multi_view[0],
            features_multi_view[0],
            matched_centres, cameras[0],
            filename=f'cam_1-reproject_frame_{frame:08}.png'
        )
        f3.utility.plot_reproject(
            images_multi_view[1],
            features_multi_view[1],
            matched_centres, cameras[1],
            filename=f'cam_2-reproject_frame_{frame:08}.png'
        )
        f3.utility.plot_reproject(
            images_multi_view[2],
            features_multi_view[2],
            matched_centres, cameras[2],
            filename=f'cam_3-reproject_frame_{frame:08}.png'
        )

for f in feature_handlers:
    f.close()

f = open('locations_3d.pkl', 'wb')
for frame in frames:
    pickle.dump(frame, f)
f.close()
