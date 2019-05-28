import sys
sys.path.append('result')
import configparser
import numpy as np
import fish_track as ft


config = ft.utility.Configure('configure.ini')

indices = config.Kernel.principle_axes
indices = indices.split(',')
indices = [int(i) for i in indices]

cluster_num = config.Kernel.cluster_number


try:
    shapes = np.load('./fish_shape_collection.npy')
except FileNotFoundError:
    shapes = np.load('result/fish_shape_collection.npy')

kernels = ft.kernel.get_kernels(shapes, indices, cluster_num)
np.save('shape_kernels', kernels)
