import re
import os
import sys
import configparser
from glob import glob
from scipy import ndimage
from util import get_updated_name
import fish_track as ft

conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

folder = conf['file']['video_folder']
filetype = conf['file']['video_format']

length = int(conf['video']['background_rolling_length'])
blur = float(conf['video']['blur'])
local = int(conf['video']['local'])
bop = int(conf['video']['binary_open_size'])
cache_method = 'mean'

for path in glob(os.path.join(folder, f"*.{filetype}")):
    filename = os.path.basename(path)
    name = get_updated_name(filename, dict(conf['rename']))
    bg_name = name + '-bg.avi'
    fg_name = name + '-fg.avi'
    bg_path = os.path.join(folder, bg_name)
    fg_path = os.path.join(folder, fg_name)

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
