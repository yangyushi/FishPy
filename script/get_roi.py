#!/usr/bin/env python3
import fish_track as ft
import configparser
import numpy as np


config = configparser.ConfigParser()
config.read('configure.ini')
data_path = config['Data']['path']
data_type = config['Data']['type']
roi = config['Process']['roi']
roi = [int(x) for x in roi.split(', ')]

if data_type == 'video':
    images = ft.read.iter_video(data_path)
elif data_type == 'images':
    images = ft.read.iter_image_sequence(data_path)
else:
    raise TypeError("Wrong data type", data_type, " Only [video] and [images] are supported")

roi = ft.measure_roi(images, roi)
roi_str = ', '.join([str(r) for r in roi])

config.set('Process', 'roi', roi_str)

with open('configure.ini', 'w') as f:
    config.write(f)
