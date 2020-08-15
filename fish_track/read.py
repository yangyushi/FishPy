#!/usr/bin/env python3
import re
import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import sparse
from collections import deque


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


def get_frame_number(file_name):
    vidcap = cv2.VideoCapture(file_name)
    return int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))


def iter_video(file_name, roi=None, start=0):
    vidcap = cv2.VideoCapture(file_name)
    vidcap.set(1, start)
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


def iter_image_sequence(folder, prefix='frame_', start=0):
    file_names = os.listdir(folder)
    image_names = [folder + '/' + fn for fn in file_names if prefix in fn]
    image_names.sort()
    for name in image_names[start:]:
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


def get_foreground_movie(video, background, output='foreground.avi', process=lambda x: x, fps=15, local=5, thresh_flag=0, bop=0):
    """
    video: name of the video to be processed (raw video recorded from camera)
    background: background video. It is assumend the foreground is *darker*, so foreground = backgroudn - video
    preprocess: a python function to process *both* foreground and background before the substraction
    fps: the frame rate of the obtained video
    local: the range of the local threshold
    threh_flag: flags for the global threshold, 0 - otsu; 1 - triangle; >1 - fixed value threshold, value is given number
    bop: the binary open morphological transformation, 0 - don't use; >0 - size of the kernel
    """
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
        binary_adpt = cv2.adaptiveThreshold(fg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, local, 0)
        binary = binary * binary_adpt
        if bop > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel=np.ones(int(bop)))
        fg[binary == 0] = 0
        out.write(fg)

    im_cap.release()
    bg_cap.release()
    out.release()


def get_frames_from_xyz(filename, ncols=3):
    f = open(filename, 'r')
    frames = []
    for line in f:
        is_head = re.match(r'(\d+)\n', line)
        if is_head:
            frames.append([])
            particle_num = int(is_head.group(1))
            f.readline()  # jump through comment line
            for j in range(particle_num):
                data = re.split(r'\s', f.readline())[1: 1 + ncols]
                frames[-1].append(list(map(float, data)))
    f.close()
    return np.array(frames)


def get_trajectories_xyz(filename):
    frames = get_frames_from_xyz(filename)
    trajs = np.moveaxis(frames, 0, 2)
    return trajs


def make_difference_movie(
        filename, start=0, end=0,
        kind='video', transform=lambda x: x):
    """
    Return a movie that stores the difference between successive frames
    """
    if kind.lower() == 'video':
        video_iter = iter_video(filename, start=start)
    elif kind.lower() == 'image':
        video_iter = iter_image_sequence(filename, start=start)
    else:
        raise TypeError("only [video] and [image] kind is supported")

    total_frame = get_frame_number(filename)
    if end == 0:
        end = total_frame
    elif end > total_frame:
        end = total_frame

    f0 = transform(next(video_iter))
    result = []
    for i in range(end - start - 1):
        f1 = transform(next(video_iter))
        diff = sparse.coo_matrix(f1 - f0)
        f0 = f1
        result.append(diff)
    return result
