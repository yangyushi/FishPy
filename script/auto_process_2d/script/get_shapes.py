#!/usr/bin/env python3
import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import configparser

if "fish_shape_collection.npy" in os.listdir('.'):
    exit(0)

import fish_track as ft


def get_shapes(image, threshold, win_size):
    """
    measure and aligh individual shapes in an image
    """
    threshold = threshold * image.max()

    fg = image * (image > threshold)

    maxima = ft.shape.get_maxima(image, threshold, win_size)

    if len(maxima) == 0:
        shapes = []
    else:
        sub_images = ft.shape.get_sub_images(fg, maxima, win_size)
        shapes = []
        for i, sub_img in enumerate(sub_images):
            if not ndimage.label(sub_img > 0)[1] < 2:
                continue

            aligned_image, aspect_ratio = ft.shape.align_sub_image(
                sub_img, want_ar=True
            )
            shapes.append(aligned_image)
    return shapes


conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

frame_start = int(conf['locate']['shape_frame_start'])
frame_end = int(conf['locate']['shape_frame_end'])

path = conf['file']['video_file']
name = os.path.basename(path)
folder = os.path.dirname(path)
vid_name = os.path.join(folder, os.path.splitext(name)[0] + '-rec.avi')

threshold = float(conf['locate']['intensity_threshold'])
win_size = int(conf['locate']['size_max'])

video = ft.read.iter_video(vid_name)

shapes = []

count = 0
for i, img in enumerate(video):
    if i < frame_start:
        continue
    s = get_shapes(img, threshold, win_size)
    shapes += s
    if i > frame_end:
        break

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
