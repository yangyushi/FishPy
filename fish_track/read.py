#!/usr/bin/env python3
import numpy as np
import cv2
import os
from collections import deque
from PIL import Image
from scipy import ndimage


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
        if count > max_frame:
            break
    return background / count


def iter_image_sequence(folder, prefix='frame_'):
    file_names = os.listdir(folder)
    image_names = [folder + '/' + fn for fn in file_names if prefix in fn]
    image_names.sort()
    for name in image_names:
        yield np.array(Image.open(name))


def get_background_movie(file_name, length=300, output='background.avi', fps=15):
    """
    Get a movie of background, being a box-average along time series with length of `length`
    Save bg every segment (25 frames), and save difference otherwise
    """
    vidcap = cv2.VideoCapture(file_name)

    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height), False)

    history = deque()
    bg_sum = np.zeros((height, width), dtype=np.uint64)

    for frame in range(frame_count):
        success, image = vidcap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        history.append(image)
        frame += 1
        bg_sum += image
        if frame == length:
            bg = (bg_sum / length).astype(np.uint8)
            for _ in range(length):
                out.write(bg)
        elif frame > length:
            bg_sum -= history.popleft()
            bg = (bg_sum / length).astype(np.uint8)
            out.write(bg)
    out.release()
    vidcap.release()


def get_foreground_movie(video, background, output='foreground.avi', process=lambda x: x, fps=15, local=5, thresh_flag=0):
    im_cap = cv2.VideoCapture(video)
    bg_cap = cv2.VideoCapture(background)
    width = int(im_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(im_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(im_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height), False)
    for frame in range(frame_count):
        success, image = im_cap.read()
        assert success, "reading video failed"
        image = process(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64))
        success, bg = bg_cap.read()
        assert success, "reading background failed"
        bg = process(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float64))
        fg = bg - image
        fg = fg - fg.min()
        fg = (fg/fg.max()*255).astype(np.uint8)
        if thresh_flag == 0:
            _, binary = cv2.threshold(fg, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif thresh_flag == 1:
            _, binary = cv2.threshold(fg, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
        elif thresh_flag > 1:
            _, binary = cv2.threshold(fg, thresh_flag, 255,cv2.THRESH_BINARY)
        fg[binary == 0] = 0
        binary_adpt = cv2.adaptiveThreshold(fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, local, 0)
        fg[binary_adpt == 0] = 0
        out.write(fg)

    im_cap.release()
    bg_cap.release()
    out.release()
