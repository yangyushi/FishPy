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

setup_sections = ['Stereo', 'Plot', 'Calibration', 'DEFAULT']

grid_size = float(config['Calibration']['grid_size'])
win_size = int(config['Calibration']['win_size'])
corner_number = tuple([int(x) for x in re.split(r'[\s,]+', config['Calibration']['corner_number']) if x])
use_int = config['Calibration'].get('use_int')
if use_int:
    use_int = bool(eval(use_int))
cameras = {}

for key in config:
    if key in setup_sections:
        continue
    camera_name = key
    folder_int = config[key]['intrinsic']


    if use_int:  # check if there is a pkl file
        use_int = glob.glob(f'{folder_int}/*.pkl')

    file_ext = config[key]['external']
    order = config[key]['order']
    image_fmt = config[key]['format']

    cam = f3.Camera()
    if use_int:
        cam_model = glob.glob(f'{folder_int}/*.pkl')[0]
        cam.read_int(cam_model)
        cam.calibrate_ext(
                ext_image=file_ext,
                grid_size=grid_size,
                win_size=(win_size, win_size),
                order=order,
                corner_number=corner_number,
                show=True 
        )
    else:
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
