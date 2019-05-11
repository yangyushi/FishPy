#!/usr/bin/env python
import glob
import pickle
import numpy as np
import configparser
import fish_3d as f3
import fish_track as ft
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

config = configparser.ConfigParser()
config.read('config.ini')

with open(f'cameras.pkl', 'rb') as f:
    cameras = pickle.load(f)

frame_start = int(config['Stereo']['frame_start'])
frame_end = int(config['Stereo']['frame_end'])
normal = (0, 0, int(config['Stereo']['normal']))
water_level = int(config['Stereo']['water_level'])
water_depth = float(config['Stereo']['water_depth'])
sample_size = int(config['Stereo']['sample_size'])
tol_2d = float(config['Stereo']['tol_2d'])
tol_3d = float(config['Stereo']['tol_3d'])

for frame in range(frame_start, frame_end):
    images_multi_view = []
    clusters_multi_view = []
    cameras_ordered = []
    
    for i, cam_name in enumerate(cameras):
        view_config = configparser.ConfigParser()
        view_config.read(config[cam_name]['config'])

        roi = view_config['Process']['roi']
        threshold = float(view_config['Locate']['img_threshold'])
        x0, y0, size_x, size_y = [int(x) for x in roi.split(', ')]
        roi = (slice(y0, y0 + size_y, None), slice(x0, x0 + size_x))

        background = np.load(config[cam_name]['background'])

        # the degree 180 is not included, it should be cuvered by another "upside-down" shape
        angle_number = int(view_config['Locate']['orientation_number'])
        angles = np.linspace(0, 180, angle_number)  

        shape_kernels = np.load(config[cam_name]['shape'])

        video_format = view_config['Data']['type']
        if video_format == 'images':
            images = ft.read.iter_image_sequence(config[cam_name]['images'])
        elif video_format == 'video':
            images = ft.read.iter_video(config[cam_name]['images'])

        f = open(config[cam_name]['feature'], 'rb')
        for _ in range(frame + 1):
            image = next(images)
            feature = pickle.load(f)
        f.close()

        img_threshold = float(view_config['Locate']['img_threshold'])

        clusters = ft.oishi.get_clusters(
            feature, background - image, shape_kernels, angles, roi,
            threshold=img_threshold, kernel_threshold=img_threshold
        )
        cameras_ordered.append(cameras[cam_name])
        images_multi_view.append(image)
        clusters_multi_view.append(clusters)
        
    # stereomatcing using refractive epipolar relationships
    matched_indices = f3.stereolink.greedy_match_centre(
        clusters_multi_view, cameras_ordered, images_multi_view,
        depth=water_depth, normal=normal, water_level=water_level,
        tol_2d=tol_2d, tol_3d=tol_3d, report=True
    )
    
    # from 2D clusters to 3D clouds
    clouds = f3.stereolink.reconstruct_clouds(
            cameras_ordered, matched_indices, clusters_multi_view,
            water_level=water_level, normal=normal, sample_size=sample_size, tol=tol_3d
    )

    # merge the overlapped 3D clouds
    clouds = f3.stereolink.merge_clouds(clouds, min_dist=tol_3d, min_num=sample_size)
    matched_centre = np.array([np.mean(cloud, 0) for cloud in clouds])

    f3.utility.plot_reproject(images_multi_view[0], matched_centre, cameras_ordered[0], filename=f'reproject_cam_1_frame_{frame:04}.pdf')
    f3.utility.plot_reproject(images_multi_view[1], matched_centre, cameras_ordered[1], filename=f'reproject_cam_2_frame_{frame:04}.pdf')
    f3.utility.plot_reproject(images_multi_view[2], matched_centre, cameras_ordered[2], filename=f'reproject_cam_3_frame_{frame:04}.pdf')

    print(f'frame {frame}', len(matched_centre))
    np.save(f'location_3d_frame_{frame:04}', matched_centre)

f = open('positions.pkl', 'wb')
frames = glob.glob(r'./location_3d_frame_*.npy')
frames.sort()
for frame in frames:
    pickle.dump(np.load(frame), f)
f.close()
