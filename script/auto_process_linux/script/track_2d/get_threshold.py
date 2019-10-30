#!/usr/bin/env python3
import os
import fish_track as ft 
import fish_gui
import configparser
import numpy as np


config = ft.utility.Configure('configure.ini')
data_path = config.Data.path
data_type = config.Data.type

if 'background.npy' in os.listdir('.'):
    bg = 'background.npy'
elif 'result' in os.listdir('.'):
    if 'background.npy' in os.listdir('result'):
        bg = 'result/background.npy'
    else:
        bg = None
else:
    bg = None

if data_type == 'video':
    images = ft.read.iter_video(data_path)
elif data_type == 'images':
    images = ft.read.iter_image_sequence(data_path)
else:
    raise TypeError("Wrong data type", data_type, " Only [video] and [images] are supported")

threshold = fish_gui.get_threshold(images, 'configure.ini', bg)

config.Fish.threshold = threshold
config.write('configure.ini')
