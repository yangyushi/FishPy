import os
import cv2
import numpy as np
import configparser
from scipy.spatial.transform import Rotation
import fish_3d as f3
import matplotlib.pyplot as plt
import pickle


has_cameras = True
fns = os.listdir(".")
for i in range(1, 4):
    has_cameras *= f"cam-{i}.json" in fns

if not has_cameras:
    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read('configure.ini')
    use_conic = eval(conf['option']['use_conic'])
    force_circle = eval(conf['option']['force_circle'])

    print("Load 2D points and 3D points")
    img_points = np.load('img_points.npy')
    n_image = img_points.shape[1]
    obj_points = np.load('obj_points.npy')
    obj_points_homo = np.concatenate((
        obj_points, np.ones((obj_points.shape[0], 1))
    ), axis=1)

    print("Load camera intrinisc parameters.")
    cameras = []
    for i in range(1, 4):
        camera_fn = conf['input'][f'cam_int_{i}']
        kind = camera_fn.split('.')[-1]
        if kind == 'pkl':
            with open(camera_fn, 'rb') as f:
                cameras.append(pickle.load(f))
        elif kind == 'json':
            camera = f3.Camera()
            camera.load_json(camera_fn)
            cameras.append(camera)
        else:
            exit(f"unsupported camera type:", kind)

    camera_triplets = []
    if use_conic:
        print("load the measured conic points")
        conic_matrices = []
        for i in range(1, 4):
            data = np.loadtxt(
                conf['input'][f"conic_{i}"], delimiter=',',
                skiprows=1, usecols=[1, 2]
            )
            data = cameras[i-1].undistort_points(data, want_uv=True)
            conic_matrices.append(
                f3.ellipse.get_conic_matrix(
                    f3.ellipse.fit_ellipse(data)
                )
            )
        print("collect cameras calibrated with pnp and c2c optimisation.")
        for i in range(n_image):
            print(f"Calibrating with image {i+1}")
            p2d = img_points[:, i, :, :]
            trip = f3.utility.get_optimised_camera_triplet_c2c(
                cameras, conic_matrices, p2d, p3d=obj_points,
                force_circle=force_circle, method='Nelder-Mead'
            )
            camera_triplets.append(trip)
    else:  # calibrate without conic optimisation
        for i in range(n_image):
            p2d = img_points[:, i, :, :]
            trip = f3.utility.get_optimised_camera_triplet(
                cameras, p2d, p3d=obj_points
            )
            camera_triplets.append(trip)

    # remove outliers
    camera_triplets = f3.utility.remove_camera_triplet_outliers(camera_triplets, threshold=3.0)
    n_triplets = len(camera_triplets)

    # average the euclidean transform between cameras
    camera_triplets_mean = f3.utility.get_cameras_with_averaged_euclidean_transform(
        camera_triplets, index=0
    )

    # output the camera with minimumn cost
    costs = np.empty(n_triplets)
    if use_conic:
        for i in range(n_triplets):
            trip = camera_triplets_mean[i]
            costs[i] = f3.utility.get_cost_camera_triplet_c2c(
                trip, conic_matrices, img_points, obj_points
            )

    else:
        for i in range(n_triplets):
            trip = camera_triplets_mean[i]
            p2d = img_points[:, i, :, :]  # n_view, n, 2
            p3d_reproj = f3.cstereo.refractive_triangulate(
                *[c.undistort_points(p) for c, p in zip(trip, p2d)],
                *[c.p for c in trip],
                *[c.o for c in trip],
            )
        costs[i] = np.mean(np.abs(p3d_reproj[:, 2]))

    print("Output Cameras")
    for i, cam in enumerate(camera_triplets_mean[np.argmin(costs)]):
        cam.save_json(f'cam-{i+1}.json')

    print("Plotting Error")
    plt.hist(costs, bins=25, histtype='step')
    if use_conic:
        plt.xlabel("Cost")
    else:
        plt.xlabel("Average Z value (mm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('calib-cost.pdf')
