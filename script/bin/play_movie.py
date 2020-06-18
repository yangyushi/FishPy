#!/usr/bin/env python3
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) != 2:
    exit("fishpy-play filename")

with open(sys.argv[1], 'rb') as f:
    movie = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(-400, 0)
ax.view_init(elev=70, azim=90)

scatter = ax.scatter([], [], [], 'o', color='lightblue', edgecolor='teal')
quiver = ax.quiver3D([], [], [], [], [], [], color='teal')

def update(frame_num):
    global quiver
    global scatter
    frame = movie[frame_num]
    if len(frame) > 0:
        v = movie.velocity(frame_num)
        quiver.remove()
        scatter.remove()
        quiver = ax.quiver3D(*frame.T, *v.T, color='teal', length=10)
        scatter = ax.scatter(*frame.T, color='lightblue', edgecolor='teal', s=10)
    dummy = ax.plot([], [])[0]
    return [dummy]

ani = FuncAnimation(fig, update, frames=range(len(movie)), blit=True, interval=10)

plt.show()
