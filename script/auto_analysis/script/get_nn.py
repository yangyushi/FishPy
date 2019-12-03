import os
import sys
import pickle
import numpy as np
import fish_corr as fc
import fish_track as ft
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


if len(sys.argv) != 5:
    print("-" * 50)
    print("Invalid amount of arguments")
    print("Correct format: python get_locations.py [trajectory_file] [output_folder] [ignore_vertices] [max_length]")
    print("-" * 50)
    sys.exit(1)

if 'tank_model.pkl' not in os.listdir('.'):
    print("-" * 50)
    print("No tank model, run get_tank.py first")
    print("-" * 50)
    sys.exit(1)

traj_file = sys.argv[1]
output_folder = sys.argv[2]
ignore_vertices = bool(int(sys.argv[3]))
max_length = int(sys.argv[4])

with open(traj_file, 'rb') as f:
    trajectories = pickle.load(f)

movie = ft.Movie(trajectories)

nn_locations = []
nn_dists_mean = []

for i in range(len(movie) - 1):
    frame = movie[i]
    indices = movie.indice_pair(i)[0]
    if len(indices) > 1 + (int(ignore_vertices) * 3):
        velocity = movie.velocity(i)[indices]
        loc, dist = fc.static.get_nn_with_velocity(frame[indices], velocity, ignore_vertices)
        nn_locations += loc.tolist()
        nn_dists_mean.append(np.mean(dist))

nn_locations = np.array(nn_locations)

plt.plot(nn_dists_mean, color='tomato', markerfacecolor='none')
plt.ylabel('Average NN distance (mm)', fontsize=14)
plt.xlabel('Time (frame)', fontsize=14)
plt.savefig(f'{output_folder}/nn_time.png')
plt.close()


# Calculate the spatial distribution of the nearest neighbours

r = max_length
bins = 41
hist, binedges = np.histogramdd(
    nn_locations, density=True,
    bins=(
        np.linspace(-r, r, bins, endpoint=True),
        np.linspace(-r, r, bins, endpoint=True),
        np.linspace(-r, r, bins, endpoint=True)
    )
)

bc = binedges[0][:-1]

plt.gcf().set_size_inches(8, 4.5)
plt.subplot(121).imshow(hist[:, :, bins//2].T)#, vmin=0, vmax=1e-5)
plt.axis('off')
plt.subplot(122).imshow(hist[:, bins//2, :].T)#, vmin=0, vmax=1e-5)
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{output_folder}/nn_dist.pdf')
plt.close()

np.save('mean_nn_movie', nn_dists_mean)
np.save('nn_locations', nn_locations)
