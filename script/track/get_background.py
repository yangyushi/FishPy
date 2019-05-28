#!/usr/bin/env python3
import sys
sys.path.append('result')
import fish_track as ft
import configparser
import numpy as np
from PIL import Image
from scipy import ndimage


config = ft.utility.Configure('configure.ini')
data_path = config.Data.path
data_type = config.Data.type

frame_max = config.Process.background_frame
step = config.Process.background_step

blur  = config.Process.gaussian_sigma

if config.Process.gaussian_sigma != 0:
    def denoise(x): return ndimage.gaussian_filter(x, blur)
else:
    def denoise(x): return x

if config.Process.normalise == 'std':
    def normalise(x): return x / x.std()
elif config.Process.normalise == 'max':
    def normalise(x): return x / x.max()
elif config.Process.normalise == 'None':
    def normalise(x): return x

if data_type == 'video':
    images = ft.read.iter_video(data_path)
elif data_type == 'images':
    images = ft.read.iter_image_sequence(data_path)
else:
    raise TypeError("Wrong data type", data_type, " Only [video] and [images] are supported")


bg = ft.read.get_background(
        images, step=step, max_frame=frame_max,
        process=lambda x: normalise(denoise(x))
        )

np.save('background', bg)

im = Image.fromarray(bg.astype(np.uint8))
im.save("background.png")
