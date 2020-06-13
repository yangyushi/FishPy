#!/usr/bin/env python3
import os
import sys
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if len(sys.argv) < 2:
    print("Plot [GReTA] result or 3D [Linking] result?")
if sys.argv[1].lower() == "linking":
    traj_file = 'link_3d/trajectories.pkl'
    if os.path.isfile(traj_file):
        with open(traj_file, 'rb') as f:
            trajs = pickle.load(f)
    else:
        exit("No trajectories from GReTA tracking found")
elif sys.argv[1].lower() == "greta":
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

lengths = [len(t[0]) for t in trajs]
plt.hist(lengths)
plt.tight_layout()
if len(sys.argv) > 2:
    plt.savefig('traj-stat.png')
    plt.show()
else:
    plt.savefig('traj-stat.png')
