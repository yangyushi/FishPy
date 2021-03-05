#!/usr/bin/env python3
import os
import re
import sys

def help():
    print()
    print("-" * 72)
    print("Getting calibration order with Deep Learning")
    print("Syntax:\tfishpy-get-calib-order [folder] [pattern] [camera_num] [corner_num]")
    print("\t(default folder: .")
    print("\t(default pattern: cam_{i}-*.tiff, i iter over camera indices)")
    print("\t(default camera_num: 3, i = [1, 2, 3])")
    print('\t(default corner_number: "23,15")')
    print("\nExample:")
    print("\tfishpy-get-calib-order calib_ext")
    print("\tfishpy-get-calib-order calib_ext camera-*.jpeg")
    print("\tfishpy-get-calib-order calib_ext camera-*.jpeg 5")
    print('\tfishpy-get-calib-order calib_ext camera-*.jpeg 5 "8,6"')
    print("-" * 72)
    print()
    exit(0)

def is_centre_valid(row, col, image, size):
    """
    Check if the sub-image is inside the full image
    """
    r_max, c_max = image.shape
    is_valid = True
    is_valid *= row + size < r_max
    is_valid *= col + size < c_max
    is_valid *= row - size >= 0
    is_valid *= col - size >= 0
    return is_valid

def get_sub_image(row, col, image, size):
    box = (
        slice(row - size, row + size),
        slice(col - size, col + size),
    )
    return image[box]

def parse_argv():
    if len(sys.argv) == 1:
            folder = "."
            pattern = "cam_{i}-*.tiff"
            cam_num = 3
            corner_number = (23, 15)
    elif len(sys.argv) == 2:
        if sys.argv[1].lower() in ["-h", "h", "help"]:
            help()
        else:
            folder = sys.argv[1]
            pattern = "cam_{i}-*.tiff"
            cam_num = 3
            corner_number = (23, 15)
    elif len(sys.argv) == 3:
        folder = sys.argv[1]
        pattern = sys.argv[2]
        cam_num = 3
        corner_number = (23, 15)
    elif len(sys.argv) == 4:
        folder = sys.argv[1]
        pattern = sys.argv[2]
        cam_num = int(sys.argv[3])
        corner_number = (23, 15)
    elif len(sys.argv) == 5:
        folder = sys.argv[1]
        pattern = sys.argv[2]
        cam_num = int(sys.argv[3])
        corner_match = re.match(r'(\d+),\s*(\d+)', sys.argv[4])
        if corner_match:
            corner_number = (
                int(corner_match.group(1)),
                int(corner_match.group(2))
            )
        else:
            help()
    else:
        help()
    return folder, pattern, cam_num, corner_number


folder, pattern, cam_num, corner_number = parse_argv()

import tensorflow as tf
from PIL import Image
from glob import glob
import numpy as np
import json
import cv2

size = 100
file_path = os.path.realpath(__file__)
bin_folder = os.path.dirname(file_path)
model =  tf.keras.models.load_model(f'{bin_folder}/calib_order_model.h5')
orders = { f'cam_{i}': []  for i in range(1, cam_num + 1) }


for i in range(1, cam_num+1):
    file_pattern = pattern.format(i=f"{i}")
    match_pattern = re.sub(r"\*", r"(\\d+)", file_pattern)
    filenames = glob(
        "{f}/{p}".format(f=folder, p=file_pattern)
    )

    filenames.sort(
        key=lambda x: int(re.search(match_pattern, x).group(1))
    )

    corner_images = []
    for fn in filenames:
        img = np.array(Image.open(fn).convert('L'))
        ret, corners = cv2.findChessboardCorners(
            img, corner_number,
            flags=sum((
                cv2.CALIB_CB_FAST_CHECK,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                ))
        )
        if ret:
            xy = corners[0]
            c = np.flip(xy.ravel().astype(int))
            if is_centre_valid(*c, img, size):
                corner_img = get_sub_image(*c, img, size)[:, :, np.newaxis]
                corner_img = corner_img - corner_img.mean()
                corner_img = corner_img / corner_img.std()
                corner_images.append(corner_img)
            else:
                exit(f"corder detection failed for {fn}")
        else:
            exit(f"corder detection failed for {fn}")
    if len(corner_images) != len(filenames):
        exit("Calibration image detection failed!")
    corner_images = np.array(corner_images)
    predictions = model.predict(corner_images)
    for j, pred in enumerate(predictions):
        if pred[0] < 0.5:
            orders[f'cam_{i}'].append('321x')
            result = '3'
        else:
            orders[f'cam_{i}'].append('x123')
            result = 'X'
        print(f"processing {filenames[j]:<20}, P={pred[0]:.4f}, result: [{result}]")

with open('calib-order.json', 'w') as f:
    json.dump(orders, f, indent=" "*4)
