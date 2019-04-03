#!/usr/bin/env python3
import sys
sys.path.append('result')
import fish_track as ft
import configparser
import numpy as np
from PIL import Image


config = configparser.ConfigParser()
config.read('configure.ini')
data_path = config['Data']['path']
data_type = config['Data']['type']

frame_max = int(config['Process']['background_frame'])
step = int(config['Process']['background_step'])


if data_type == 'video':
    images = ft.read.iter_video(data_path)
elif data_type == 'images':
    images = ft.read.iter_image_sequence(data_path)
else:
    raise TypeError("Wrong data type", data_type, " Only [video] and [images] are supported")

bg = ft.read.get_background(images, step=step, max_frame=frame_max)

np.save('background', bg)

im = Image.fromarray(bg.astype(np.uint8))
im.save("background.png")
