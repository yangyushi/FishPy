import fish_track as ft
from scipy import ndimage
import sys
import os

name = sys.argv[1]
length = 150
blur = 2

if name not in os.listdir('.'):
    print(f"file not found ({name}), exit")
else:
    bg_name = os.path.splitext(name)[0] + '-bg.avi'
    fg_name = os.path.splitext(name)[0] + '-fg.avi'
    ft.get_background_movie(
            name, length=length, output=bg_name
            )
    ft.get_foreground_movie(
            name, bg_name, fg_name,
            process=lambda x: ndimage.gaussian_filter(x, blur)
            )
