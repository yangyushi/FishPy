#!/usr/bin/env python3
import os
import re
import json
import pickle
import fish_3d as f3
import configparser

config = configparser.ConfigParser()
config.read('configure.ini')

setup_sections = ['PostProcess', 'Stereo', 'Temporal', 'Calibration', 'DEFAULT']

grid_size = float(config['Calibration']['grid_size'])
win_size = int(config['Calibration']['win_size'])
corner_number = tuple([int(x) for x in re.split(r'[\s,]+', config['Calibration']['corner_number']) if x])

with open(config['Calibration']['order_file'], 'r') as f:
    calib_orders_json = json.load(f)

cameras = []
calib_files = []
calib_orders = []
camera_names = []

for cam_name in config:
    if cam_name in setup_sections:
        continue
    orders = calib_orders_json[cam_name]
    folder = config['Calibration']['folder']
    suffix = config['Calibration']['format']
    filenames = [f'{folder}/{cam_name}-{i+1}.{suffix}' for i in range(len(orders))]

    cam = f3.Camera()
    cam.read_int(config[cam_name]['intrinsic'])

    cameras.append(cam)
    calib_files.append(filenames)
    calib_orders.append(orders)
    camera_names.append(cam_name)

is_calibrated = True
for name, cam in zip(camera_names, cameras):
    is_calibrated *= f'{name}.pkl' in os.listdir('.')

if not is_calibrated:
    f3.camera.calib_mult_ext(
        *cameras, *calib_files, *calib_orders,
        grid_size, corner_number, win_size
    )
    for name, cam in zip(camera_names, cameras):
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(cam, f)
