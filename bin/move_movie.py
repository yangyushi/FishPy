#!/usr/bin/env python3
"""
move the movie file to the database (Ghibli folder)
"""
import re
import os
import sys
from glob import glob
from shutil import copyfile

pwd = os.getcwd()
ghibli = '/home/yy17363/Dropbox/Ivan/Ghibli'


if len(sys.argv) == 3:
    sub_folder = sys.argv[1]
    movie_type = sys.argv[2]
elif len(sys.argv) == 2 and sys.argv[1] in ['h', '-h', 'help', '-help']:
    info = "fishpy-move-movie "
    info += "[folder, default: auto_process_] "
    info += "[type, default: linking]\n"
    exit(info)
elif len(sys.argv) == 2:
    sub_folder = sys.argv[1]
    movie_type = "linking"
else:
    sub_folder = "auto_process_"
    movie_type = "linking"


try:
    date = re.search('2\d{7}', pwd).group()
except AttributeError:
    exit("Current Working Directory lacks the date")

for folder in glob(f'{pwd}/{sub_folder}*'):
    idx = re.match(f'{pwd}/{sub_folder}(\d+)', folder).group(1)
    source = f'{folder}/movie-{movie_type}.pkl'
    target = f'{ghibli}/movie-{date}-vid-{idx}-{movie_type}.pkl'
    if os.path.basename(source) in os.listdir(folder):
        copyfile(source, target)
    else:
        print(f"No {movie_type} movie file in {folder}")
