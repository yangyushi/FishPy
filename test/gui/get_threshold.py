#!/usr/bin/env python3
import os
import fish_track as ft 
from PIL import Image
import gui
import configparser
import numpy as np


config = ft.utility.Configure('configure.ini')
bg = 'background.npy'

image = np.array(Image.open('cam-1.tiff'))
images = [image, image]

threshold = gui.get_threshold(iter(images), 'configure.ini', bg)

config.Fish.threshold = threshold
config.write('configure.ini')
