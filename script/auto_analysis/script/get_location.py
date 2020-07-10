import os
import sys
import pickle
import numpy as np
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
traj_name = os.path.basename(traj_file).split('.')[0]
output_folder = sys.argv[2]

with open(traj_file, 'rb') as f:
    trajectories = pickle.load(f)

with open('tank_model.pkl', 'rb') as f:
    tank = pickle.load(f)

traj_location_file = traj_name + '-locations.npy'
want_calculate = traj_location_file not in os.listdir('.')


if want_calculate:
    movie = fc.Movie(trajectories, blur=0, interpolate=False)
    locations = [] # shape is (frame, n, 2)
    for i, frame in enumerate(movie):
        if len(frame) > 0:
            locations += frame.tolist()
    locations = np.array(locations)
    np.save(traj_location_file, locations)
else:
    locations = np.load(traj_location_file)

radii, angles, distances = tank.get_curvilinear(locations)

plt.hist(angles/np.pi*180, histtype='step', bins=100, color='teal', density=True, label='Angle')
plt.xlabel("Angle (degree)", fontsize=14)
plt.ylabel("PDF", fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_folder}/angle_distribution.pdf")
plt.close()

plt.hist(distances, histtype='step', bins=50, density=True, label='fish-tank distance', color='tomato')
plt.hist(radii, histtype='step', bins=50, density=True, label='projection radii', color='teal')
plt.legend(fontsize=14)
plt.xlabel('Length / mm', fontsize=14)
plt.ylabel('PDF', fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_folder}/curvilinear_distribution.pdf")
plt.close()


X, Y, Z = tank.get_xyz(locations)
hist_2d, x_edges, y_edges  = np.histogram2d(
    np.sqrt(X**2 + Y**2), Z, bins=(50, 50), range=((0, 700), (0, 700)),
    density=True
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Distribution Spatial (RZ)")
ax.imshow(hist_2d.T, vmin=0, vmax=None)
ax.set_ylim(0, 50)
ax.set_xticks(range(0, 50, 5))
ax.set_xticklabels(range(0, 700, 50))
ax.set_yticks(range(0, 50, 5))
ax.set_yticklabels(range(0, 700, 50))
ax.set_xlabel("X")
ax.set_ylabel("Y")
r = np.arange(1000)
ax.plot(r/700*50, tank.z(r)/700*50, color='w', ls='--', lw=2)
ax.set_xlim(0, 50)
plt.savefig(f'{output_folder}/spatial_distribution_rz.pdf')
plt.close()

hist_2d, x_edges, y_edges  = np.histogram2d(
    X, Y, bins=(50, 50), range=((-700, 700), (-700, 700)),
    density=True
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Distribution Spatial (XY)")
ax.imshow(hist_2d.T, vmin=0, vmax=None)
ax.set_ylim(0, 50)
ax.set_xticks(range(0, 50, 5))
ax.set_xticklabels(range(-700, 700, 50))
ax.set_yticks(range(0, 50, 5))
ax.set_yticklabels(range(-700, 700, 50))
ax.set_xlabel("R")
ax.set_ylabel("Z")
r = np.arange(1000)
ax.set_xlim(0, 50)
plt.savefig(f'{output_folder}/spatial_distribution_xy.pdf')
plt.close()
