#!/usr/bin/env python3
from PIL import Image
import fish_track as ft 
import fish_gui
import configparser
import numpy as np


config = ft.utility.Configure('configure.ini')
roi = config.Process.roi
roi = [int(x) for x in roi.split(', ')]

image = np.array(Image.open('cam-1.tiff'))
images = [image, image]

roi = fish_gui.measure_roi(iter(images), roi)
roi_str = ', '.join([str(r) for r in roi])

config.Process.roi = roi_str
config.write('configure.ini')
