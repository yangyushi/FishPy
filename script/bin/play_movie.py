#!/usr/bin/env python3
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) < 2:
    exit("fishpy-play Filename [Arrow Length] [Delay]")

with open(sys.argv[1], 'rb') as f:
    movie = pickle.load(f)

if len(sys.argv) == 3:
    length = float(sys.argv[2])
    delay = 10
elif len(sys.argv) == 4:
    length = float(sys.argv[2])
    delay = float(sys.argv[3])
else:
    length = 10
    delay = 10


xmax = movie[0].T[0].max()
ymax = movie[0].T[1].max()
zmax = movie[0].T[2].max()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)
ax.set_zlim(-zmax, zmax)
ax.view_init(elev=70, azim=90)

scatter = ax.scatter([], [], [], 'o', color='lightblue', edgecolor='teal')
quiver = ax.quiver3D([], [], [], [], [], [], color='teal')

def update(frame_num):
    global quiver
    global scatter
    frame = movie[frame_num] - movie[frame_num].mean(axis=0)
    if len(frame) > 0:
        v = movie.velocity(frame_num)
        quiver.remove()
        scatter.remove()
        quiver = ax.quiver3D(*frame.T, *v.T, color='teal', length=length)
        scatter = ax.scatter(*frame.T, color='lightblue', edgecolor='teal', s=10)
    dummy = ax.plot([], [])[0]
    return [dummy]

ani = FuncAnimation(fig, update, frames=range(len(movie)), blit=True, interval=delay)

plt.show()
