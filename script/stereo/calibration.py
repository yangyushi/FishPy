#!/usr/bin/env python3
import pickle
import numpy as np
import re
import cv2
import glob
import matplotlib.pyplot as plt
import fish_3d as f3 
import fish_track as ft
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

setup_sections = ['Stereo', 'Calibration', 'DEFAULT']

grid_size = float(config['Calibration']['grid_size'])
win_size = int(config['Calibration']['win_size'])
corner_number = tuple([int(x) for x in re.split(r'[\s,]+', config['Calibration']['corner_number']) if x])
cameras = {}

for key in config:
    if key in setup_sections:
        continue
    camera_name = key
    folder_int = config[key]['intrinsic']
    file_ext = config[key]['external']
    order = config[key]['order']
    image_fmt = config[key]['format']

    cam = f3.Camera()

    cam.calibrate(
            int_images=[fn for fn in glob.glob(f'{folder_int}/*.{image_fmt}')],
            ext_image=file_ext,
            grid_size=grid_size,
            win_size=(win_size, win_size),
            order=order,
            corner_number=corner_number,
            show=True
    )
    cameras.update({camera_name: cam})
    with open(f'{camera_name}.pkl', 'wb') as f:
        pickle.dump(cam, f)

with open(f'cameras.pkl', 'wb') as f:
    pickle.dump(cameras, f)
