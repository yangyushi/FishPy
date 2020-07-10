#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
from scipy.stats import binned_statistic_2d, binned_statistic
import fish_corr as fc
import matplotlib.pyplot as plt


if len(sys.argv) < 3:
    print("-" * 50)
    print("Invalid amount of arguments")
    print("Correct format: python get_locations.py [trajectory_file] [output_folder]")
    print("-" * 50)
    sys.exit(1)

if 'tank_model.pkl' not in os.listdir('.'):
    print("-" * 50)
    print("No tank model, run get_tank.py first")
    print("-" * 50)
    sys.exit(1)

traj_file = sys.argv[1]
output_folder = sys.argv[2]

with open(traj_file, 'rb') as f:
    trajs = pickle.load(f)

with open('tank_model.pkl', 'rb') as f:
    tank = pickle.load(f)

if 'movie.pkl' in os.listdir('.'):
    with open('movie.pkl', 'rb') as f:
        movie = pickle.load(f)
else:
    movie = fc.Movie(trajs, blur=1, interpolate=False)
    with open('movie.pkl', 'wb') as f:
        pickle.dump(movie, f)

speed_multi_frames = []
speed_average = []
X = []
Y = []
Z = []
D = []  # distance from fish to tank

projs_xyz = []  # projected coordinates
velocities_multi_frames = []

for frame in range(len(movie)-1):
    indices = movie.indice_pair(frame)[0]
    if len(indices) == 0:
        continue
    positions = movie[frame][indices]
    velocities = movie.velocity(frame)[indices]  # shape of velocity is n, 3
    speed = np.linalg.norm(velocities, axis=1)
    speed_multi_frames += speed.tolist()
    speed_average.append(np.mean(speed))
    x, y, z = tank.get_xyz(positions)
    proj = tank.get_projection(positions).T
    _, _, d = tank.get_curvilinear(positions)
    X += x.tolist()
    Y += y.tolist()
    Z += z.tolist()
    D += d.tolist()
    projs_xyz += proj.tolist()
    velocities_multi_frames += velocities.tolist()


"""
The distribution of speed
"""

plt.hist(speed_multi_frames, bins=100, histtype='step', color='teal', density=True)
plt.xlabel("Speed (mm / frame)", fontsize=14)
plt.ylabel("PDF", fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_folder}/speed_distribution.pdf')
plt.close()

"""
The average speed at different frames
"""

plt.plot(speed_average, color='teal')
plt.xlabel("Time (frame)", fontsize=14)
plt.ylabel("<Speed (mm/frame)>", fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_folder}/speed_movie.png')
plt.close()


"""
Spatial Distribution of speed, 1D
"""

hist_speed_distance, bin_edges, _ = binned_statistic(D, speed_multi_frames, bins=50)
bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
plt.plot(bin_centres, hist_speed_distance)
plt.xlabel("Distance to tank (mm)", fontsize=14)
plt.ylabel("<Speed> in each bin", fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_folder}/speed_distance_stat.pdf')
plt.close()

"""
Spatial Distribution of speed, 2D
"""

bin_num = 20
bin_x = np.linspace(-700, 700, bin_num)
bin_y = np.linspace(-700, 700, bin_num)
bin_z = np.linspace(0, 400, bin_num)

hist_2d_speed_xy, bin_edges_x, bin_edges_y, _ = binned_statistic_2d(X, Y, speed_multi_frames, bins=(bin_x, bin_y))
hist_2d_speed_xz, _, _, _ = binned_statistic_2d(X, Z, speed_multi_frames, bins=(bin_x, bin_z))


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

im_xy = axs[0].imshow(hist_2d_speed_xy.T)
axs[0].set_title("Speed Distribution (XY)")
axs[0].set_ylim(-1, bin_num+1)
axs[0].axis('off')
fig.colorbar(im_xy, ax=axs[0])

im_xz = axs[1].imshow(hist_2d_speed_xz.T)
axs[1].set_title("Speed Distribution (XZ)")
axs[1].set_ylim(-1, bin_num+1)
axs[1].axis('off')
fig.colorbar(im_xz, ax=axs[1])

plt.savefig(f"{output_folder}/speed_spatial_dist.pdf")
plt.close()


"""
Calculate the alignment between velocity & tank
"""

proj_vecs =  projs_xyz - np.vstack((X, Y, Z)).T
perpendicular_vecs = proj_vecs / np.vstack(np.linalg.norm(proj_vecs, axis=1))
velocity_vecs = velocities_multi_frames / np.vstack(np.linalg.norm(velocities_multi_frames, axis=1))
alignments = np.array([1 - abs(pv @ vv) for pv, vv in zip(perpendicular_vecs, velocity_vecs)])


"""
    Spatial Distribution of velocity alignment, 1D
"""

hist_align_distance, bin_edges, _ = binned_statistic(D, alignments, bins=50)
bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
plt.plot(bin_centres, hist_align_distance)
plt.xlabel("Distance to tank (mm)", fontsize=14)
plt.ylabel("<Alignment to Wall> in each bin", fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_folder}/alignment_distance_stat.pdf')
plt.close()


"""
    Spatial Distribution of alignment, 2D
"""

bin_num = 20
bin_x = np.linspace(-700, 700, bin_num)
bin_y = np.linspace(-700, 700, bin_num)
bin_z = np.linspace(0, 400, bin_num)

hist_2d_alignment_xy, bin_edges_x, bin_edges_y, _ = binned_statistic_2d(X, Y, alignments, bins=(bin_x, bin_y))
hist_2d_alignment_xz, _, _, _ = binned_statistic_2d(X, Z, alignments, bins=(bin_x, bin_z))


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

im_xy = axs[0].imshow(hist_2d_alignment_xy.T)
axs[0].set_title("Alignment Distribution (XY)")
axs[0].set_ylim(-1, bin_num+1)
axs[0].axis('off')
fig.colorbar(im_xy, ax=axs[0])

im_xz = axs[1].imshow(hist_2d_alignment_xz.T)
axs[1].set_title("Alignment Distribution (XZ)")
axs[1].set_ylim(-1, bin_num+1)
axs[1].axis('off')
fig.colorbar(im_xz, ax=axs[1])

plt.savefig(f"{output_folder}/alignemnt_spatial_dist.pdf")
plt.close()

"""
    fit the speed distribution with Maxwell Boltzmann distribution
"""

bins = np.linspace(0, np.nanmax(speed_multi_frames), 51)

dimension, (bin_centres, spd_pdf), (fit_x, fit_y) = fc.utility.fit_maxwell_boltzmann(np.array(speed_multi_frames), bins)

plt.scatter(
    bin_centres, spd_pdf, color='tomato', facecolor='w',
    label=f'Dimension: {dimension:.2f}',
    zorder=1
)

plt.plot(
    fit_x, fit_y, color='k',
    zorder=2, ls='--', alpha=0.8,
    label='Maxwell Boltzmann fit',
)

plt.legend(fontsize=14)
plt.gca().set_yscale('log')
plt.ylim(np.min(spd_pdf[spd_pdf > 0])/10, np.max(spd_pdf)*10)
plt.xlabel('Speed (mm/frame)', fontsize=14)
plt.ylabel('PDF(Speed)', fontsize=14)
plt.savefig(f"{output_folder}/speed_fit.pdf")
plt.close()

with open('movie.pkl', 'wb') as f:
    pickle.dump(movie, f)
