#!/usr/bin/env python
import os
import glob
import pickle
import numpy as np
import configparser
import fish_3d as f3
import fish_track as ft
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

config = ft.utility.Configure('config.ini')

if 'cameras.pkl' in os.listdir('.'):
    with open(f'cameras.pkl', 'rb') as f:
        cameras = pickle.load(f)
elif 'cameras.pkl' in os.listdir('result'):
    with open(f'result/cameras.pkl', 'rb') as f:
        cameras = pickle.load(f)
else:
    raise FileNotFoundError("Can't locate calibrated cameras, run make calib")

frame_start = config.Stereo.frame_start
frame_end = config.Stereo.frame_end
normal = (0, 0, config.Stereo.normal)
water_level = config.Stereo.water_level
water_depth = config.Stereo.water_depth
sample_size = config.Stereo.sample_size
tol_2d = config.Stereo.tol_2d
tol_3d = config.Stereo.tol_3d
see_cluster = bool(config.Plot.see_cluster)
see_reprojection = bool(config.Plot.see_reprojection)

for frame in range(frame_start, frame_end):
    images_multi_view = []
    rois_multi_view = []
    features_multi_view = []
    clusters_multi_view = []
    cameras_ordered = []

    for i, cam_name in enumerate(cameras):
        cam_config = eval(f'config.{cam_name}')
        view_config = ft.utility.Configure(cam_config.config)

        roi = view_config.Process.roi
        x0, y0, size_x, size_y = [int(x) for x in roi.split(', ')]
        roi = (slice(y0, y0 + size_y, None), slice(x0, x0 + size_x))

        # the degree 180 is not included, it should be covered by another "upside-down" shape
        angle_number = view_config.Locate.orientation_number
        angles = np.linspace(0, 180, angle_number)

        shape_kernels = np.load(cam_config.shape)

        video_format = view_config.Data.type
        if video_format == 'images':
            images = ft.read.iter_image_sequence(cam_config.images)
        elif video_format == 'video':
            images = ft.read.iter_video(cam_config.images)

        f = open(cam_config.feature, 'rb')

        for _ in range(frame + 1):
            image = next(images)
            feature = pickle.load(f)
        f.close()

        fg = image[roi]

        clusters = ft.oishi.get_clusters(
            feature, shape_kernels, angles, roi,
            kernel_threshold=config.Stereo.kernel_threshold
        )


        cameras_ordered.append(cameras[cam_name])
        images_multi_view.append(image)
        clusters_multi_view.append(clusters)
        rois_multi_view.append(roi)
        features_multi_view.append(feature)

        if see_cluster:
            plt.imshow(fg, cmap='gray')
            for c in clusters:
                plt.scatter(*c.T, alpha=0.5)
            plt.show()

    # stereomatcing using refractive epipolar relationships
    matched_indices = f3.stereolink.greedy_match_centre(
        clusters_multi_view, cameras_ordered, images_multi_view,
        depth=water_depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, tol_3d=tol_3d, report=True, sample_size=sample_size
    )

    # from 2D clusters to 3D clouds
    clouds, err = f3.stereolink.reconstruct_clouds(
            cameras_ordered, matched_indices, clusters_multi_view,
            water_level=water_level, normal=normal, sample_size=sample_size, tol=tol_3d
    )

    matched_centre = np.array([np.mean(cloud, 0) for cloud in clouds])

    if see_reprojection:
        f3.utility.plot_reproject(
            images_multi_view[0],
            rois_multi_view[0], features_multi_view[0],
            matched_centre, cameras_ordered[0],
            filename=f'cam_1-reproject_frame_{frame:04}.png'
        )
        f3.utility.plot_reproject(
            images_multi_view[1],
            rois_multi_view[1], features_multi_view[1],
            matched_centre, cameras_ordered[1],
            filename=f'cam_2-reproject_frame_{frame:04}.png'
        )
        f3.utility.plot_reproject(
            images_multi_view[2],
            rois_multi_view[2], features_multi_view[2],
            matched_centre, cameras_ordered[2],
            filename=f'cam_3-reproject_frame_{frame:04}.png'
        )
    print(f'frame {frame}', len(matched_centre))
    np.save(f'location_3d_frame_{frame:04}', matched_centre)

f = open('positions.pkl', 'wb')
frames = glob.glob(r'./location_3d_frame_*.npy')
frames.sort()
for frame in frames:
    pickle.dump(np.load(frame), f)
f.close()
