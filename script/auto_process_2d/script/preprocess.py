import re
import os
import sys
import fish_track as ft
from scipy import ndimage
import configparser

conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

path = conf['file']['video_file']
name = os.path.basename(path)
folder = os.path.dirname(path)

length = int(conf['video']['background_rolling_length'])
blur = float(conf['video']['blur'])
local = int(conf['video']['local'])
bop = int(conf['video']['binary_open_size'])
cache_method = 'mean'

if name not in os.listdir(folder):
    print(f"file not found ({name}), exit")
else:
    bg_name = os.path.splitext(name)[0] + '-bg.avi'
    fg_name = os.path.splitext(name)[0] + '-fg.avi'
    bg_path = folder + '/' + bg_name
    fg_path = folder + '/' + fg_name

    if bg_name not in os.listdir(folder):
        ft.get_background_movie(
                path, length=length, output=bg_path, cache=cache_method
                )

    if fg_name not in os.listdir(folder):
        ft.get_foreground_movie(
                path, bg_path, fg_path,
                process=lambda x: ndimage.gaussian_filter(x, blur),
                local=local, bop=bop
                )
