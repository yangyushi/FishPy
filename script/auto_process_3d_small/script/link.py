import os
import sys
import pickle
import numpy as np
import configparser
import matplotlib.pyplot as plt

import fish_track as ft
import fish_corr as fc

conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

linker_name = conf['link']['linker']
linker_range = float(conf['link']['range'])
dx_max = int(conf['link']['relink_range'])
dt_max = int(conf['link']['relink_time'])
blur = int(conf['link']['relink_blur'])
threshold = int(conf['link']['threshold'])
relink_window = int(conf['link']['relink_window'])

filename = 'locations_3d.pkl'

# detect total frames
frame_number = 0
with open(filename, 'rb') as f:
    while True:
        try:
            pickle.load(f)
            frame_number += 1
        except EOFError:
            break

if 'vanilla_trajs.pkl' not in os.listdir("."):
    frames = []
    with open(filename, 'rb') as f:
        for _ in range(0, frame_number):
            frames.append(pickle.load(f))

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
                max_frame=frame_number,
                dx=dx,
                dt=dt_max,
                blur_velocity=blur,
                debug=False
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

    plt.figure(figsize=(5, 5))
    for t in movie.trajs:
        plt.plot(*t.positions.T[:2], marker='+')
    plt.xlabel("X / mm")
    plt.ylabel("Y / mm")
    plt.tight_layout()
    plt.savefig('trajectories.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    lengths = []
    for t in movie.trajs:
        lengths.append(t.positions.shape[0])
    plt.hist(lengths, bins=25, color='teal')
    plt.xlabel("Length / frame")
    plt.ylabel("PDF")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('trajectories-length.png')
    plt.close()
