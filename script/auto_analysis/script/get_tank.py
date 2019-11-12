#!/usr/bin/env python3
import json
import pickle
import numpy as np
import fish_3d as f3
import fish_corr as fc
import sys
import matplotlib.pyplot as plt


if len(sys.argv) not in [3, 4]:
    print("-" * 50)
    print("Invalid amount of arguments")
    print("Correct format: python get_tank.py [tank_centre.json] [input_folder] show")
    print("-" * 50)
    sys.exit(1)

cameras = []
input_folder = sys.argv[2]
for cam_name in [f'{input_folder}/cam_{i+1}.pkl' for i in range(3)]:
    with open(cam_name, 'rb') as f:
        cameras.append(pickle.load(f))
        cameras[-1].update()

centres = []
with open(sys.argv[1], "r") as f:
    tank_centres = json.load(f)
for i in range(3):
    centres.append(tank_centres[f'cam_{i+1}'])
centres = np.array(centres)

# undistort these centres
centres_undistort = np.empty(centres.shape)
for i, cam in enumerate(cameras):
    centres_undistort[i] = cam.undistort(centres[i], want_uv=True)
centres_undistort = np.array(centres_undistort)

# calculate the 3D position of the centre
centre_3d, error = f3.ray_trace.ray_trace_refractive_faster(centres_undistort, cameras, z=0)
print(f"Maximum reprojection error is {error:.4f} pixels")

tank = fc.static.Tank(centre_3d)
with open('tank_model.pkl', 'wb') as f:
    pickle.dump(tank, f)

if len(sys.argv) > 3:
    from PIL import Image

    fig, ax = plt.subplots(2, 2)
    ax[0][0].scatter(*f3.ray_trace.reproject_refractive_no_distort(centre_3d, cameras[0]), marker='+', color='tomato')
    ax[0][0].scatter(*centres[0], color='tomato', facecolor='none')
    ax[0][0].imshow(Image.open('../images/cam-1.png'))

    ax[0][1].scatter(*f3.ray_trace.reproject_refractive_no_distort(centre_3d, cameras[1]), marker='+', color='tomato')
    ax[0][1].scatter(*centres[1], color='tomato', facecolor='none')
    ax[0][1].imshow(Image.open('../images/cam-2.png'))

    ax[1][0].scatter(*f3.ray_trace.reproject_refractive_no_distort(centre_3d, cameras[2]), marker='+', color='tomato')
    ax[1][0].scatter(*centres[2], color='tomato', facecolor='none')
    ax[1][0].imshow(Image.open('../images/cam-3.png'))

    ax[1][1].axis('off')

    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.set_size_inches(12, 9)
    fig.tight_layout()
    plt.savefig('tank_base.png')
    plt.close()
