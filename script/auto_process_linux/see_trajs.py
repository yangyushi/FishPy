#!/usr/bin/env python3
import sys
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

with open('link_3d/trajectories.pkl', 'rb') as f:
    trajs = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for t in trajs:
    ax.plot(*t[1].T)
fig.tight_layout()
if len(sys.argv) > 1:
    plt.savefig('trajs.png')
    plt.show()
else:
    plt.savefig('trajs.png')
