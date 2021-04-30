import re
import os
import cv2
import glob
import pickle
import numpy as np
import configparser
from PIL import Image
import matplotlib.pyplot as plt

import fish_3d as f3
import fish_track as ft


def get_length_unit(image, L, H, corner_number, grid_size):
    """
    Get the ratio between mm / pixel, used to convert image coordinates
        to real coordinates.

    Args:
        filename (str): the path to an image of the chessboard.
        L (int): the side length of the output image, the output image
            should have a square shape.
        H (np.ndarray): the homography to similarly rectify the image.
        corner_number (tuple): the number of corner points on the image
        grid_size (float): the side length of a square grid on the
            chessboard, in the unit milimeters

    Return:
        tuple: state (True or False) and the ratio, if the corner detection
            failed the state will be False and the ratio will be nan
    """
    img_obj = Image.fromarray(image)
    img_obj.save('sample-original.png')

    image = cam.undistort_image(image)
    img_obj = Image.fromarray(image)
    img_obj.save('sample-undistorted.png')

    image = cv2.warpPerspective(image, H, (L, L))
    img_obj = Image.fromarray(image)
    img_obj.save('sample-rectified.png')

    ret, corners = cv2.findChessboardCorners(
        image, corner_number,
        flags=sum((
            cv2.CALIB_CB_FAST_CHECK,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            ))
    )
    if ret:
        corners = cv2.cornerSubPix(
            image, corners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        )

        img = cv2.imread('sample-rectified.png')
        img = cv2.drawChessboardCorners(
            img, corner_number, corners, ret
        )
        cv2.imwrite('sample-rectified.png', img)
        dist_mm = grid_size * np.sqrt(
            (corner_number[0] - 1)**2 + (corner_number[1] - 1)**2
        )
        dist_px = np.linalg.norm(corners[0] - corners[-1])
        return True, dist_mm / dist_px
    else:
        False, np.nan


# load parameters
conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

fps = float(conf['video']['fps'])
path = conf['file']['video_file']
name = os.path.basename(path)
folder = os.path.dirname(path)
calib_folder = conf['file']['calib_folder']
calib_format = conf['camera']['calib_format']

grid_size = float(conf['camera']['grid_size'])
corner_number = tuple([
    int(x) for x in re.split(
        r'[\s,]+', conf['camera']['corner_number']
    ) if x
])


# calculate the homography
H = []
with open('cameras.pkl', 'rb') as f:
    cameras = pickle.load(f)
for cam in cameras:
    H.append(f3.utility.get_homography(cam))
H = np.mean(H, axis=0)

# calculate the ratio to convert the unit
for sample_name in glob.glob(os.path.join(calib_folder, f"*.{calib_format}")):
    sample = np.array(Image.open(sample_name).convert('L'))
    L = np.max(sample.shape) + 100
    is_success, unit_ratio = get_length_unit(
        sample, L, H, corner_number, grid_size
    )
    if is_success:
        break

conf['camera']['mm/px'] = str(unit_ratio)
with open('configure.ini', 'w') as f:
    conf.write(f)

# generate the rectified video
fg_name = os.path.join(folder, os.path.splitext(name)[0] + '-fg.avi')
output = os.path.join(folder, os.path.splitext(name)[0] + '-rec.avi')
if os.path.isfile(output):
    exit(0)

cap = cv2.VideoCapture(fg_name)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output, fourcc, fps, (L, L), False)

for frame in range(frame_count):
    success, image = cap.read()
    assert success, "reading video failed"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cam.undistort_image(image)
    image = cv2.warpPerspective(image, H, (L, L))
    out.write(image)

cap.release()
out.release()
