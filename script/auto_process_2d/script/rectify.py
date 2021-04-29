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

fg_name = os.path.join(folder, os.path.splitext(name)[0] + '-fg.avi')
output = os.path.join(folder, os.path.splitext(name)[0] + '-rec.avi')

if os.path.isfile(output):
    exit(0)

H = []
with open('cameras.pkl', 'rb') as f:
    cameras = pickle.load(f)
for cam in cameras:
    H.append(f3.utility.get_homography(cam))
H = np.mean(H, axis=0)

sample_fn = glob.glob(os.path.join(calib_folder, f"*.{calib_format}"))[0]
sample = np.array(Image.open(sample_fn).convert('L'))
img_obj = Image.fromarray(sample)
img_obj.save('sample-original.png')

L = np.max(sample.shape) + 100
sample = cam.undistort_image(sample)
img_obj = Image.fromarray(sample)
img_obj.save('sample-undistorted.png')

sample = cv2.warpPerspective(sample, H, (L, L))
img_obj = Image.fromarray(sample)
img_obj.save('sample-rectified.png')

ret, corners = cv2.findChessboardCorners(
    sample, corner_number,
    flags=sum((
        cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        ))
)
if ret:
    corners = cv2.cornerSubPix(
        sample, corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    )

    img = cv2.imread('sample-rectified.png')
    img = cv2.drawChessboardCorners(
        img, corner_number, corners, ret
    )
    cv2.imwrite('sample-rectified.png', img)
else:
    raise RuntimeError("Corner detection failed!")

# update the pixel size information
dist_mm = grid_size * np.sqrt(
    (corner_number[0] - 1)**2 + (corner_number[1] - 1)**2
)
dist_px = np.linalg.norm(corners[0] - corners[-1])
conf['camera']['mm/px'] = str(dist_mm / dist_px)
with open('configure.ini', 'w') as f:
    conf.write(f)

# generate the rectified video
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
