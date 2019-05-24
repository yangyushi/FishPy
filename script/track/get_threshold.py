#!/usr/bin/env python3
import os
import fish_track as ft 
import gui
import configparser
import numpy as np


config = configparser.ConfigParser()
config.read('configure.ini')
data_path = config['Data']['path']
data_type = config['Data']['type']
mvd_path = config['Data']['mvd']

if 'background.npy' in os.listdir('.'):
    bg = 'background.npy'
elif 'result' in os.listdir('.'):
    if 'background.npy' in os.listdir('result'):
        bg = 'result/background.npy'
    else:
        bg = None
else:
    bg = None

config_mvd = configparser.ConfigParser()
config_mvd.read(mvd_path)

fish_mvd = ft.shape.MVD(mvd_path)

if data_type == 'video':
    images = ft.read.iter_video(data_path)
elif data_type == 'images':
    images = ft.read.iter_image_sequence(data_path)
else:
    raise TypeError("Wrong data type", data_type, " Only [video] and [images] are supported")

threshold = gui.get_threshold(images, fish_mvd.intensity.threshold, bg)

config_mvd.set('intensity', 'threshold', str(threshold))

with open(mvd_path, 'w') as f:
    config_mvd.write(f)
