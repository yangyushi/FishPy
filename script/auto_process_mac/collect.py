import sys
import os
from glob import glob
from shutil import copy

target_folder = sys.argv[1]
os.mkdir(f'{target_folder}/details-cam-1')
os.mkdir(f'{target_folder}/details-cam-2')
os.mkdir(f'{target_folder}/details-cam-3')
os.mkdir(f'{target_folder}/config')

files = [f'track_2d/features_2d-cam_{i+1}.pkl' for i in range(3)]
files += [f'track_2d/shapes_2d-cam_{i+1}.npy' for i in range(3)]
files += [f'track_3d/cam_{i+1}.pkl' for i in range(3)]
files.append('track_3d/locations_3d.pkl')
for f in files:
    copy(f, target_folder)

files_config = [f'track_2d/cam-{i+1}/configure.ini' for i in range(3)]
files_config.append('track_3d/configure.ini')
for i, f in enumerate(files_config):
    if i < 3:
        copy(f, f'{target_folder}/config/config_2d-cam_{i+1}.ini')
    else:
        copy(f, f'{target_folder}/config/config_3d.ini')

for i in range(3):
    files = glob(f'track_2d/cam-{i+1}/result/*.pdf')
    files += glob(f'track_2d/cam-{i+1}/result/*.npy')
    for f in files:
        copy(f, f'{target_folder}/details-cam-{i+1}')
sys.exit(0)
