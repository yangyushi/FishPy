#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from .refraction_fix import get_poi, get_trans_vec, get_intersect_of_lines


def triangulate_3point(x1, x2, x3, p1, p2, p3):
    M = np.zeros((9, 7))
    M[:3, :4] = p1
    M[3:6, :4] = p2
    M[6:, :4] = p3
    M[:3, 4] = -x1
    M[3:6, 5] = -x2
    M[6:, 6] = -x3
    u, s, vh = np.linalg.svd(M)
    X = vh[-1, :4]
    return X / X[3]

def triangulate_in_water(positions, cameras, water_level, frame=0):

    fish = positions_gt.T[frame]
    assert len(positions) == len(cameras)

    lines = []
    camera_index = 0
    fig, ax = plt.subplots(2, 1)

    for position, camera in zip(positions, cameras):
        camera_index += 1
        camera_origin = -1 * camera.r.T @ camera.t  # (x, y, z)
        poi = get_poi(camera.get_projection_matrix(), water_level, position[:2])
        i = poi - camera_origin
        t = get_trans_vec(i)
        n = np.array([0, 0, 1])
        a1 = np.arccos(i @ n / np.linalg.norm(n) / np.linalg.norm(i))
        a2 = np.arccos(t @ n / np.linalg.norm(n) / np.linalg.norm(t))
        assert abs((np.sin(a1) / np.sin(a2)) - 1.33) < 1e-6
        line = {'point': poi, 'unit': t}
        lines.append(line)

        ax[0].scatter(poi[1], poi[0], color='w', label=f'on water (camera {camera_index})', edgecolor=cm.Dark2(camera_index))
        ax[0].scatter(camera_origin[1], camera_origin[0], color=cm.Dark2(camera_index), label=f'camera {camera_index}')
        ax[0].plot([poi[1], fish[1]], [poi[0], fish[0]], color=cm.Dark2(camera_index), linewidth=1)
        ax[0].plot([camera_origin[1], poi[1]], [camera_origin[0], poi[0]], color=cm.Dark2(camera_index), linewidth=1)
        ax[0].set_xlim(-700, 800)
        ax[0].set_ylim(-500, 1000)

        ax[1].scatter(poi[1], poi[2], color='w', label=f'on water (camera {camera_index})', edgecolor=cm.Dark2(camera_index))
        ax[1].scatter(camera_origin[1], camera_origin[2], color=cm.Dark2(camera_index), label=f'camera {camera_index}')
        ax[1].plot([poi[1], fish[1]], [poi[2], fish[2]], color=cm.Dark2(camera_index), linewidth=1)
        ax[1].plot([camera_origin[1], poi[1]], [camera_origin[2], poi[2]], color=cm.Dark2(camera_index), linewidth=1)
        ax[0].set_xlim(-700, 800)
        ax[1].set_ylim(-500, 1000)

    ax[0].scatter(fish[1], fish[0], color='w', edgecolor='k', label='fish')
    ax[1].scatter(fish[1], fish[2], color='w', edgecolor='k', label='fish')
    ax[0].legend()
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.tight_layout()
    fig.set_size_inches(6, 12)

    plt.savefig(f'../results/fix_refraction/camera_poi_frame{frame:02}.pdf')
    plt.close()
    return get_intersect_of_lines(lines)
