import os
import sys
import pickle
import numpy as np
import configparser
import matplotlib.pyplot as plt

import fish_track as ft
import fish_corr as fc

if 'movie.pkl' in os.listdir('.'):
    exit(0)

filename = "features.pkl"
conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

linker_name = conf['link']['linker']

frame_start = int(conf['link']['frame_start'])
frame_end = int(conf['link']['frame_end'])
linker_range = float(conf['link']['link_range'])
dx_max = int(conf['link']['relink_range'])
dt_max = int(conf['link']['relink_time'])
blur = int(conf['link']['relink_blur'])
threshold = int(conf['link']['threshold'])
relink_window = int(conf['link']['relink_window'])
ratio_px_to_mm = float(conf['camera']['mm/px'])

# detect total frames
if frame_end == 0:
    with open(filename, 'rb') as f:
        while True:
            try:
                pickle.load(f)
                frame_end += 1
            except EOFError:
                break

frame_number = frame_end - frame_start

if 'vanilla_trajs.pkl' not in os.listdir("."):
    frames = []
    with open(filename, 'rb') as f:
        for _ in range(frame_start):
            pickle.load(f)
        for _ in range(0, frame_number):
            # shape of oishi features: (6, n)
            pos_xy = pickle.load(f)[:2].T
            pos_xy *= ratio_px_to_mm
            frames.append(pos_xy)

    if linker_name.lower() == 'trackpy':
        linker = ft.TrackpyLinker(linker_range, 0)
    elif linker_name.lower() == 'active':
        linker = ft.ActiveLinker(linker_range)
    else:
        print("Invalid linker: ", linker_name)

    vanilla_trajs = linker.link(frames)
    vanilla_trajs = [t for t in vanilla_trajs if len(t[0]) > 1]

    with open('vanilla_trajs.pkl', 'wb') as f:
        pickle.dump(vanilla_trajs, f)
else:
    with open(f'vanilla_trajs.pkl', 'rb') as f:
        vanilla_trajs = pickle.load(f)

if f'trajectories.pkl' not in os.listdir('.'):
    if len(vanilla_trajs) > 1:
        trajs = ft.relink(vanilla_trajs, 1, 1, blur_velocity=blur)
        for dx in range(2, dx_max + 2):
            trajs = ft.relink_by_segments(
                trajs,
                window_size=relink_window,
                max_frame=frame_end,
                dx=dx,
                dt=dt_max,
                blur_velocity=blur
            )
        trajs = [t for t in trajs if len(t[0]) > threshold]
    else:
        trajs = vanilla_trajs

    with open('trajectories.pkl', 'wb') as f:
              pickle.dump(trajs, f)


    movie = fc.Movie(trajs, blur=None, interpolate=True)
    movie.make()
    movie.save_xyz('movie.xyz')

    with open('movie.pkl', 'wb') as f:
        pickle.dump(movie, f)

    plt.figure(figsize=(8, 8))
    for t in movie.trajs:
        plt.plot(*t.positions.T, marker='+')
    plt.xlabel("X / mm")
    plt.ylabel("Y / mm")
    plt.tight_layout()
    plt.savefig('trajectories.png')
