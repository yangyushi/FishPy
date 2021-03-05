"""
This script process a movie file to remove the static background
"""
import fish_track as ft
from scipy import ndimage
import sys
import os

name = sys.argv[1]
length = 3000
blur = 12

if name not in os.listdir('.'):
    print(f"file not found ({name}), exit")
else:
    bg_name = os.path.splitext(name)[0] + '-bg.avi'
    fg_name = os.path.splitext(name)[0] + '-fg.avi'
    if bg_name not in os.listdir('.'):
        ft.get_background_movie(
                name, length=length, output=bg_name
                )
    ft.get_foreground_movie(
            name, bg_name, fg_name,
            process=lambda x: ndimage.gaussian_filter(x, blur),
            local=21, bop=4
            )
