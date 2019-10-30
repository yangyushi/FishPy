import fish_track as ft
from scipy import ndimage
import sys
import re
import os

path = sys.argv[1]
name = os.path.basename(path)
folder = os.path.dirname(path)

length = int(sys.argv[2])
blur = int(sys.argv[3])
local = int(sys.argv[4])
bop = int(sys.argv[5])

if name not in os.listdir(folder):
    print(f"file not found ({name}), exit")
else:
    bg_name = os.path.splitext(name)[0] + '-bg.avi'
    fg_name = os.path.splitext(name)[0] + '-fg.avi'
    bg_path = folder + '/' + bg_name
    fg_path = folder + '/' + fg_name

    if bg_name not in os.listdir(folder):
        ft.get_background_movie(
                path, length=length, output=bg_path
                )

    if fg_name not in os.listdir(folder):
        ft.get_foreground_movie(
                path, bg_path, fg_path,
                process=lambda x: ndimage.gaussian_filter(x, blur),
                local=local, bop=bop
                )
