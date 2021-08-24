#!/usr/bin/env python3
"""
This script try to detect calibration board for all the images
    from multiple cameras.
The images for which the detection failed would be moved to a
    folder named "bad_img"
The remaining images will be renamed so that their indices were
    consecutive
"""
import re
import os
import sys
import cv2
import shutil
from glob import glob
from PIL import Image
import numpy as np

folder = "."
pattern = "cam_{i}-*.tiff"
cam_num = 3
corner_number = (23, 15)


indices_to_delete = []

"""
Check which images are not suitable for the auto corner detection of opencv
"""
for i in range(1, cam_num+1):
    file_pattern = pattern.format(i=f"{i}")
    match_pattern = re.sub(r"\*", r"(\\d+)", file_pattern)
    filenames = glob(
        "{f}/{p}".format(f=folder, p=file_pattern)
    )

    filenames.sort(
        key=lambda x: int(re.search(match_pattern, x).group(1))
    )

    for fn in filenames:
        img = np.array(Image.open(fn).convert('L'))

        ret, corners = cv2.findChessboardCorners(
            img, corner_number,
            flags=sum((
                cv2.CALIB_CB_FAST_CHECK,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                ))
        )

        if not ret:
            print(f"Corner detection failed for {fn}")
            index = re.search(match_pattern, fn).group(1)
            indices_to_delete.append(int(index))

indices_to_delete = set(indices_to_delete)

"""
Move the images with detection failure into a bad_img folder
"""
if "bad_img" not in os.listdir(folder):
    os.mkdir(f"{folder}/bad_img")
for idx in indices_to_delete:
    for ic in range(1, cam_num+1):
        file_pattern = pattern.format(i=ic)
        file_name = f"{folder}/{file_pattern}".replace("*", str(idx))
        shutil.move(file_name, f"{folder}/bad_img")

"""
Rename the remaining files so that their indices were consecutive
"""
for i in range(1, cam_num+1):
    file_pattern = pattern.format(i=f"{i}")
    match_pattern = re.sub(r"\*", r"(\\d+)", file_pattern)
    filenames = glob(
        "{f}/{p}".format(f=folder, p=file_pattern)
    )
    filenames.sort(
        key=lambda x: int(re.search(match_pattern, x).group(1))
    )
    for j, old_name in enumerate(filenames):
        new_name = str(file_pattern.replace("*", str(j+1)))
        os.rename(src=old_name, dst=new_name)
