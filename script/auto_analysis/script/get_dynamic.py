import os
import sys
import pickle
import numpy as np
import fish_corr as fc
import matplotlib.pyplot as plt


if len(sys.argv) != 10:
    print("-" * 50)
    print("Invalid amount of arguments")
    print("Correct format: python get_locations.py [trajectory_file] [output_folder] [fps] [body_length]")
    print("-" * 50)
    sys.exit(1)

if 'tank_model.pkl' not in os.listdir('.'):
    print("-" * 50)
    print("No tank model, run get_tank.py first")
    print("-" * 50)
    sys.exit(1)


traj_file = sys.argv[1]
output_folder = sys.argv[2]
fps = float(sys.argv[3])
body_length = float(sys.argv[4])

frame_start = int(sys.argv[5])
frame_stop = int(sys.argv[6])
bins = np.linspace(int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9])+1)
bin_width = (bins[1:] - bins[:-1])[0]

with open(traj_file, 'rb') as f:
    trajectories = pickle.load(f)

if 'movie.pkl' in os.listdir('.'):
    with open('movie.pkl', 'rb') as f:
        movie = pickle.load(f)
else:
    movie = fc.Movie(trajectories, blur=1, interpolate=False)
    with open('movie.pkl', 'wb') as f:
        pickle.dump(movie, f)

if 'dynamic.pkl' in os.listdir('.'):
    with open('dynamic.pkl', 'rb') as f:
        dyn = pickle.load(f)
else:
    dyn = fc.dynamic.Critic(movie)
    with open('dynamic.pkl', 'wb') as f:
        pickle.dump(dyn, f)

t_orders, r_orders, d_orders = dyn.get_orders(start=frame_start, stop=frame_stop)

"""
Time evolution of order parameters
"""

x = np.arange(-1, len(d_orders), 1)
y1 = np.ones(x.shape) * -1
y2 = np.ones(x.shape) * 0
y3 = np.ones(x.shape) * 1
plt.plot(x, y1, '--', color='k', alpha=0.5)
plt.plot(x, y2, '--', color='k', alpha=0.5)
plt.plot(x, y3, '--', color='k', alpha=0.5)

plt.plot(np.arange(len(d_orders))/fps, d_orders, label=r'$\Lambda$',
         color='deepskyblue', linewidth=1)
plt.plot(np.arange(len(t_orders))/fps, np.abs(t_orders), label=r'$\Phi$',
            color='tomato', linewidth=1)
plt.plot(np.arange(len(r_orders))/fps, np.abs(r_orders), label=r'$R$',
         color='darkgreen', linewidth=1)

plt.legend(fontsize=14, loc='lower right', ncol=3)
plt.ylim(-1.1, 1.1)
plt.xlim(0, len(d_orders) / fps)
plt.xlabel('time / s', fontsize=14)
plt.ylabel(f'Order Parameter', fontsize=14)
plt.gcf().set_size_inches(6, 3)
plt.tight_layout()
plt.savefig(f'{output_folder}/order_parameters_movei.png')
plt.close()

"""
Distribution of order parameters
"""

plt.hist(t_orders, histtype='step', bins=np.arange(0, 1, 0.02), density=True, label=r"Translation $\Phi$")
plt.hist(np.abs(r_orders), histtype='step', bins=np.arange(0, 1, 0.02), density=True, label=r"Rotation $R$")
plt.hist(np.abs(d_orders), histtype='step', bins=np.arange(0, 1, 0.02), density=True, label=r"Dilation $\Lambda$")
plt.legend(fontsize=14)
plt.savefig(f'{output_folder}/order_parameters_distribution.pdf')
plt.close()


"""
Correlation calculation
"""


corr, bin_edges, _ = dyn.get_corr_flctn(
    start=frame_start, stop=frame_stop, bins=bins,
    transform="T"
)

bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
plt.plot(
    bin_centres, corr, '-o', label="non-translational fluctuation",
    color='tomato', markerfacecolor='w'
)

corr, bin_edges, _ = dyn.get_corr_flctn(
    start=frame_start, stop=frame_stop, bins=bins,
    transform="I"
)

bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
plt.plot(
    bin_centres, corr, '-o', label="non-isometric fluctuation",
    color='teal', markerfacecolor='w'
)

corr, bin_edges, _ = dyn.get_corr_flctn(
    start=frame_start, stop=frame_stop, bins=bins,
    transform="S"
)
bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
plt.plot(
    bin_centres, corr, '-o', label="non-similar fluctuation",
    color='orchid', markerfacecolor='w'
)
plt.xlim(bins[0] - bin_width, bins[-1] + bin_width)
plt.plot((bins[0] - bin_width, bins[-1] + bin_width), (0, 0), color='k', linewidth=1)
plt.xlabel("r (mm)")
plt.ylabel("C(r)")
plt.legend()
plt.savefig(f"{output_folder}/corr.pdf")
plt.close()


with open('dynamic.pkl', 'wb') as f:
    pickle.dump(dyn, f)

with open('movie.pkl', 'wb') as f:
    pickle.dump(movie, f)
