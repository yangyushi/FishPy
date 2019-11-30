#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

with open('movie.pkl', 'rb') as f:
    movie = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1000, 1000)
ax.set_ylim(-1000, 1000)
ax.set_zlim(-400, 0)
ax.view_init(elev=75, azim=90)

scatter = ax.plot([], [], [], 'o', color='lightblue', markeredgecolor='teal')[0]

def update(frame):
    if len(frame) > 0:
        scatter.set_data(*frame.T[:2])
        scatter.set_3d_properties(frame[:,-1])
    return [scatter]

ani = FuncAnimation(fig, update, frames=movie, blit=True, interval=1)

plt.show()
