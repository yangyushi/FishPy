#!/usr/bin/env python3
import fish_track as ft
import fish_3d as f3
import numpy as np
import pickle
import argparse
import os
import shutil

CACHE = '.2d_movie_cache'

parser = argparse.ArgumentParser(
    description='Generate rectified 2D movie object from calibrated camera and oishi features'
)
parser.add_argument(
    "feature", type=str, help="path to the feature file"
)
parser.add_argument(
    "cam", type=str, help="path to the camera file"
)
parser.add_argument(
    "-o", "--output", type=str, default="movie.pkl",
    help="output file name"
)
parser.add_argument(
    "-f", "--frame", type=int, help="number of the frames to process"
)
parser.add_argument(
    "-r", "--range", type=int,
    help="linking range of the active linker"
)
parser.add_argument(
    "-dx", "--max_dist", type=int,
    help="maximum distance for relinking"
)
parser.add_argument(
    "-dt", "--max_time", type=int, default=10,
    help="maximum lag time for relinking"
)
parser.add_argument(
    "-m", "--min", type=int, default=1,
    help="minimal length of a valid trajectory"
)

args = parser.parse_args()

if not args.max_dist:
    args.max_dist = args.range

frames = args.frame
min_length = args.min
cam_file = args.cam
feature_file = args.feature

with open(cam_file, 'rb') as f:
    cam = pickle.load(f)
H_sim = f3.utility.get_homography(cam)

if CACHE not in os.listdir('.'):
    os.mkdir(CACHE)

# Retrieve Positions
if 'positions.pkl' in os.listdir(CACHE):
    with open(f'{CACHE}/positions.pkl', 'rb') as f:
        positions = pickle.load(f)
else:
    f = open(feature_file, 'rb')
    positions = []
    for _ in range(frames):
        xy = pickle.load(f)[:2]
        xyh = np.vstack((xy, np.ones(xy.shape[1])))
        xyh_sim = H_sim @ xyh
        xy_sim = (xyh_sim / xyh_sim[-1, :])[:2]
        positions.append(xy_sim.T)
    f.close()
    with open(f'{CACHE}/positions.pkl', 'wb') as f:
        pickle.dump(positions, f)

# Get Trajectories
linker = ft.ActiveLinker(args.range)

if 'vanilla_trajs.pkl' in os.listdir(CACHE):
    with open(f'{CACHE}/vanilla_trajs.pkl', 'rb') as f:
        vanilla_trajs = pickle.load(f)
else:
    vanilla_trajs = linker.link(positions)
    with open(f'{CACHE}/vanilla_trajs.pkl', 'wb') as f:
        pickle.dump(vanilla_trajs)

if 'trajs.pkl' in os.listdir(CACHE):
    with open(f'{CACHE}/trajs.pkl', 'rb') as f:
        trajs = pickle.load(f)
else:
    trajs = ft.relink(vanilla_trajs, 1, 1, blur=1)
    for dt in np.arange(0, args.max_time+2, 2):
        for dx in np.arange(0, args.max_dist+2, 2):
            trajs = ft.relink(trajs, dx, dt, blur=0)
    trajs = [t for t in trajs if len(t['time']) > min_length]
    with open(f'{CACHE}/trajs.pkl', 'wb') as f:
        pickle.dump(trajs, f)

# Make movie
movie = ft.Movie(trajs, blur=None, interpolate=True)
movie.make()
movie.save(args.output)

shutil.rmtree('.2d_movie_cache')
