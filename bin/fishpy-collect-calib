#!/usr/bin/env python3
"""
Collect calibration files in separate folders, like the following one.

calib-ext
├── cam-1
│   ├── Basler_acA2040-...._142218251_0001.tiff
│   └── Basler_acA2040-...._142218251_0002.tiff
├── cam-2
│   ├── Basler_acA2040-...._142221861_0001.tiff
│   └── Basler_acA2040-...._142221861_0002.tiff
└── cam-3
    ├── Basler_acA2040-...._142226049_0001.tiff
    └── Basler_acA2040-...._142226049_0002.tiff

And copy the images into a more "canonical" form, like the following one.

calib-ext/
├── cam_1-0.tiff
├── cam_1-1.tiff
├── cam_2-0.tiff
├── cam_2-1.tiff
├── cam_3-0.tiff
└── cam_3-1.tiff

The origional files were archived into file "{archive_name}.zip"
"""

import os
import shutil
from glob import glob


cam_num = 3
folder_pattern = "cam-{i}"
suffix = "tiff"
archive_name = 'original-calib-imgs'

for i in range(1, cam_num+1):
    file_pattern = os.path.join(
        folder_pattern.format(i=i),
        f"*.{suffix}"
    )
    filenames = glob(file_pattern)
    filenames.sort()
    for j, fn in enumerate(filenames):
        shutil.copy(fn, f"cam_{i}-{j+1}.{suffix}")

if archive_name in os.listdir('.'):
    os.rmdir(archive_name)
os.mkdir(archive_name)
for i in range(1, cam_num+1):
    folder_name = folder_pattern.format(i=i)
    shutil.move(
        folder_name,
        f'{archive_name}/{folder_name}'
    )
shutil.make_archive(archive_name, 'zip', archive_name)
shutil.rmtree(archive_name)
