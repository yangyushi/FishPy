#!/usr/bin/env python3
import os
import numpy as np
import fish_track as ft
import matplotlib.pyplot as plt
import configparser
from util import get_shapes


conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

folder = conf['file']['video_folder']

frame_start = int(conf['locate']['shape_frame_start'])
frame_end = int(conf['locate']['shape_frame_end'])

for name in conf['rename']:

    if f'{name}-fish-shape-collection.npy' in os.listdir('.'):
        continue

    vid_name = os.path.join(folder, name + '-fg.avi')
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

    plt.savefig(f'{name}-segment-result.pdf')
    np.save(f'{name}-fish-shape-collection', shapes)
