#!/usr/bin/env python3
import sys
sys.path.append('result')
import numpy as np
from scipy import ndimage 
import matplotlib.pyplot as plt
import fish_track as ft
import configparser

config = configparser.ConfigParser()
config.read('configure.ini')

data_path = config['Data']['path']
mvd_path = config['Data']['mvd']
data_type = config['Data']['type']

roi = config['Process']['roi']
x0, y0, size_x, size_y = [int(x) for x in roi.split(', ')]
roi = (slice(y0, y0 + size_y, None), slice(x0, x0 + size_x))

blur  = float(config['Process']['gaussian_sigma'])
frame_start = int(config['Process']['shape_frame_start'])
frame_end = int(config['Process']['shape_frame_end'])

if data_type == 'video':
    images = ft.read.iter_video(data_path)
elif data_type == 'images':
    images = ft.read.iter_image_sequence(data_path)
else:
    raise TypeError("Wrong data type", data_type)


fish_mvd = ft.shape.MVD(mvd_path)

try:
    bg = np.load('background.npy')
except FileNotFoundError:
    bg = np.load('result/background.npy')

shapes = []
volumes = []
aspect_ratios = []

for i, img in enumerate(images):
    if i < frame_start:
        continue
    for_tracking = bg - img
    for_tracking = for_tracking[roi]
    for_tracking = ndimage.gaussian_filter(for_tracking, blur)
    s, v, ar = ft.shape.get_shapes(for_tracking, fish_mvd, report=True)
    shapes += s
    volumes += v
    aspect_ratios += ar
    if i > frame_end:
        break

fig, ax = plt.subplots(2, 1)
ax[0].hist(volumes, bins=51, histtype='step')
ax[0].set_title("Intensity Volume Distribution")
ax[0].set_xlabel("sum(I)")
ax[1].hist(aspect_ratios, bins=51, histtype='step')
ax[1].set_title("Aspect Ratio Distribution")
ax[1].set_xlabel("Aspect Ratio")
plt.tight_layout()
plt.savefig('dist_intensity.pdf')

for_plot = np.random.shuffle(shapes)

number = max(int(np.sqrt(len(shapes))), 2)
number = min(number, 4)

fig, ax = plt.subplots(number, number)
for i, a in enumerate(ax.ravel()):
    a.imshow(shapes[i])
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
plt.savefig('segment_result_selection.pdf')

np.save('fish_shape_collection', shapes)
