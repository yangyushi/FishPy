#!/usr/bin/env python3
import numpy as np
import cv2
import os
from PIL import Image

def get_frame(file_name, frame):
    """
    return the frame content as an array
    """
    vidcap = cv2.VideoCapture(file_name)
    success, count = 1, 0
    while success:
        success, image = vidcap.read()
        if (count == frame) and success:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        count += 1
    return None


def iter_video(file_name, roi=None):
    vidcap = cv2.VideoCapture(file_name)
    success = 1
    while success:
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            yield image


def get_background(video_iter, step=1, max_frame=1000, process=lambda x:x):
    background = next(video_iter).astype(np.float64)
    count = 1
    for i, frame in enumerate(video_iter):
        if (i % step) == 0:
            if process:
                background += process(frame)
            else:
                background += frame
            count += 1
            if count % (max_frame / 10) == 0:
                print("X", end="")
        if count > max_frame:
            break
    print("")
    return background / count


def iter_image_sequence(folder, prefix='frame_'):
    file_names = os.listdir(folder)
    image_names = [folder + '/' + fn for fn in file_names if prefix in fn]
    image_names.sort()
    for name in image_names:
        yield np.array(Image.open(name))
