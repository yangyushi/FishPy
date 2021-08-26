#!/usr/bin/env python3
import os
import re
import cv2
import json
import glob
import pickle
import numpy as np
import configparser
import matplotlib.pyplot as plt

import fish_3d as f3
import fish_track as ft

if 'cameras.pkl' not in os.listdir('.'):
    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read('configure.ini')

    grid_size = float(conf['camera']['grid_size'])
    calib_folder = conf['file']['calib_folder']
    calib_format = conf['camera']['calib_format']
    calib_int = conf['file']['cam_internal']
    corner_number = tuple([
        int(x) for x in re.split(
            r'[\s,]+', conf['camera']['corner_number']
        ) if x
    ])

    cameras = []
    for img_fn in glob.glob(os.path.join(calib_folder, f"*.{calib_format}")):
        cam = f3.Camera()
        cam.read_int(calib_int)
        try:
            cam.calibrate_ext(
                img_fn, grid_size=grid_size,
                corner_number=corner_number, show=False
            )
        except RuntimeError:
            continue
        cameras.append(cam)

    with open('cameras.pkl', 'wb') as f:
        pickle.dump(cameras, f)
