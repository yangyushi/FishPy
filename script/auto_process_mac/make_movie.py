#!/usr/bin/env python3
import os
import sys
import fish_corr as fc
import pickle


if len(sys.argv) < 2:
    print("make movie with [GReTA] or 3D [Linking] trajectoreis?")
method = sys.argv[1].lower()

if method == "linking":
    traj_file = 'link_3d/trajectories.pkl'
    if os.path.isfile(traj_file):
        with open(traj_file, 'rb') as f:
            trajs = pickle.load(f)
    else:
        exit("No trajectories from 3D linking found")
elif method == "greta":
    traj_file = 'track_greta/trajectories.pkl'
    relink_file = 'track_greta/relinked.pkl'
    if os.path.isfile(relink_file):
        with open(relink_file, 'rb') as f:
            trajs = pickle.load(f)
    elif os.path.isfile(traj_file):
        with open(traj_file, 'rb') as f:
            trajs = pickle.load(f)
    else:
        exit("No trajectories from GReTA tracking found")
else:
    exit("Choose [GReTA] or [Linking] (case insensitive)")

if sys.argv == 3:
    blur = float(sys.argv[2])
else:
    blur = 0

movie = fc.Movie(trajs, blur=blur, interpolate=False)
movie.make()

with open(f'movie-{method}.pkl', 'wb') as f:
    pickle.dump(movie, f)
