import os
import cv2
import numpy as np
import configparser
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


    print("Load 2D points and 3D points")
    img_points = np.load('img_points.npy')
    obj_points = np.load('obj_points.npy')
    obj_points_homo = np.concatenate((
        obj_points, np.ones((obj_points.shape[0], 1))
    ), axis=1)
    force_circle = bool(eval(conf['parameter']['force_circle']))

    print("Load camera intrinisc parameters.")
    cameras = []
    for i in range(1, 4):
        with open(conf['input'][f'cam_int_{i}'], 'rb') as f:
            cameras.append(pickle.load(f))

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
    n_image = img_points.shape[1]
    camera_triplets = []
    for i in range(n_image):
        print(f"Calibrating with image {i+1}")
        p2d = img_points[:, i, :, :]
        trip = f3.utility.get_optimised_camera_triplet_c2c(
            cameras, conic_matrices, p2d, p3d=obj_points,
            force_circle=force_circle, method='Nelder-Mead'
        )
        camera_triplets.append(trip)


    camera_triplets_mean = f3.utility.get_cameras_with_averaged_euclidean_transform(
        camera_triplets, index=0
    )

    mean_z_error_vals = np.empty(n_image)
    for i in range(n_image):
        trip = camera_triplets_mean[i]
        p2d = img_points[:, i, :, :]  # n_view, n, 2
        p3d_reproj = f3.cstereo.refractive_triangulate(
            *[c.undistort_points(p) for c, p in zip(trip, p2d)],
            *[c.p for c in trip],
            *[c.o for c in trip],
        )
        mean_z_error_vals[i] = np.mean(np.abs(p3d_reproj[:, 2]))

    print("Output Cameras")
    for i, cam in enumerate(camera_triplets_mean[np.argmin(mean_z_error_vals)]):
        cam.save_json(f'cam-{i+1}.json')

    print("Plotting Error")
    plt.hist(mean_z_error_vals, bins=25, histtype='step')
    plt.xlabel("Average Z value (mm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('chessobard-z.pdf')
