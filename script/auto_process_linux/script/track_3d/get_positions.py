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

movies = []
cameras = []
camera_configurations = []
feature_handlers = []
for i, cam_name in enumerate(camera_dict):
    cam_config = eval(f'config.{cam_name}')
    images = ft.read.iter_video(cam_config.images)
    cam = camera_dict[cam_name]
    cameras.append(cam)
    movies.append(images)
    camera_configurations.append(cam_config)
    feature_handlers.append(open(cam_config.feature, 'rb'))

for frame in range(frame_start, frame_end):
    images_multi_view = []
    features_multi_view = []
    clusters_multi_view = []

    for i, cam in enumerate(cameras):
        cam_config = camera_configurations[i]

        # the degree 180 is not included, it should be covered by another "upside-down" shape
        angle_number = cam_config.orientation_number
        angles = np.linspace(0, 180, angle_number)

        shape_kernels = np.load(cam_config.shape)

        image = next(movies[i])

        feature = pickle.load(feature_handlers[i])

        clusters = ft.oishi.get_clusters(
            feature, shape_kernels, angles,
            kernel_threshold=config.Stereo.kernel_threshold
        )
        clusters = [cam.undistort_points(c, want_uv=True) for c in clusters]  # undistort each cluster

        images_multi_view.append(image)
        clusters_multi_view.append(clusters)
        features_multi_view.append(feature)

    # stereomatcing using refractive epipolar relationships
    matched_indices, matched_centres, reproj_errors = f3.three_view_cluster_match(
        clusters_multi_view, cameras,
        tol_2d=tol_2d, sample_size=sample_size, depth=water_depth,
        report=False, normal=normal, water_level=water_level
    )


    matched_indices, matched_centres, reproj_errors = f3.remove_conflict(
        matched_indices, matched_centres, reproj_errors
    )

    while True:  # try to match all un-matched flusters
        extra_indices, extra_centres, extra_reproj_errors = f3.extra_three_view_cluster_match(
            matched_indices, clusters_multi_view, cameras,
            tol_2d=tol_2d, sample_size=sample_size, depth=water_depth,
            report=False, normal=normal, water_level=water_level
        )
        if len(extra_indices) == 0:
            break

        matched_indices = np.concatenate((matched_indices, extra_indices))
        matched_centres = np.concatenate((matched_centres, extra_centres))
        reproj_errors = np.concatenate((reproj_errors, extra_reproj_errors))

        matched_indices, matched_centres, reproj_errors = f3.remove_conflict(
            matched_indices, matched_centres, reproj_errors
        )

    print(f'frame {frame}', len(matched_centres))

    if len(matched_indices) == 0:
        np.save(f'location_3d_frame_{frame:08}', np.empty((0, 3)))
        continue

    np.save(f'locations_3d/frame_{frame:08}', matched_centres)

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
frames = glob.glob(r'locations_3d/frame_*.npy')
frames.sort()
for frame in frames:
    pickle.dump(np.load(frame), f)
f.close()
