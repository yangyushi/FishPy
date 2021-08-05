import re
import cv2
import json
import os
import numpy as np
import fish_3d as f3
from glob import glob
import configparser

has_img = 'img_points.npy' in os.listdir('.')
has_obj = 'obj_points.npy' in os.listdir('.')
if (has_img and has_obj):
    exit(0)


conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')


calib_folder = conf['input']['calib_folder']
corner_number = tuple([
    int(n) for n in re.split(
        r'[,\s]', conf['parameter']['corner_number']
    ) if n
])
grid_size = float(conf['parameter']['grid_size'])
win_size = tuple([
    int(n) for n in re.split(
        r'[,\s]', conf['parameter']['win_size']
    ) if n
])

with open(f'{calib_folder}/calib-order.json', 'r') as f:
    orders = json.load(f)

img_points = [[], [], []]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

obj_points = f3.camera.get_points_from_order(
    corner_number, order='x123'
) * grid_size

for i in range(1, 4):
    match_pattern = f"cam_{i}-(\\d+).tiff"
    filenames = glob(f"{calib_folder}/cam_{i}-*.tiff")
    filenames.sort(
        key=lambda x: int(re.search(match_pattern, x).group(1))
    )
    for j, fn in enumerate(filenames):
        print(f"Processing {fn}")
        img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
        _, corners = cv2.findChessboardCorners(
            img, corner_number,
            flags=sum((
                cv2.CALIB_CB_FAST_CHECK, cv2.CALIB_CB_ADAPTIVE_THRESH
            ))
        )
        corners = cv2.cornerSubPix(img, corners, win_size, (-1, -1), criteria)
        if orders[f'cam_{i}'][j] == 'x123':
            img_points[i-1].append(np.squeeze(corners))
        else:
            img_points[i-1].append(np.squeeze(corners)[::-1])

obj_points = np.array(obj_points)
img_points = np.array(img_points)

np.save('obj_points', obj_points)
np.save('img_points', img_points)
