import numpy as np
from camera import view_1, view_2, view_3
import matplotlib.pyplot as plt
import skvideo.io
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

unit = 1000  # m --> mm

calib_chessboard_location = np.load('calibrate_origin_position.npy') * unit
chessboard_size = np.load('calibrate_origin_size.npy') * unit
grid_size = chessboard_size / 7
origin_location = calib_chessboard_location - grid_size * 2.5

# ground truth
positions_gt = np.load('fish_positions.npy')
positions_gt = positions_gt * unit
# move from blender origin to calibrate origin
positions_gt = positions_gt - np.array([origin_location]).T

p1 = view_1.get_projection_matrix()
p2 = view_2.get_projection_matrix()
p3 = view_3.get_projection_matrix()
proj_mats = [p1, p2, p3]

v1_vac = skvideo.io.vread('../videos/fish_1_camera_1.mkv', outputdict={"-pix_fmt": "gray"})
v2_vac = skvideo.io.vread('../videos/fish_1_camera_2.mkv', outputdict={"-pix_fmt": "gray"})
v3_vac = skvideo.io.vread('../videos/fish_1_camera_3.mkv', outputdict={"-pix_fmt": "gray"})

frames = list(map(np.squeeze, [v1_vac, v2_vac, v3_vac]))

def show_frames(view=0, num=3):
    for i, p in enumerate(positions_gt.T[:num]):
        img = frames[view][i]
        ph = np.hstack([p, 1])  # position in homogeneous coordinates
        p2dh = proj_mats[view] @ ph.T
        p2d = (p2dh / p2dh[-1])[:-1]
        plt.scatter(p2d[0], p2d[1], color='r', marker='+')
        plt.imshow(img)
        plt.show()

def show_traj():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*positions_gt, '-o')
    plt.show()

if __name__ == "__main__":
    show_frames(view=0, num=30)
    show_frames(view=1, num=30)
    show_frames(view=2, num=30)
