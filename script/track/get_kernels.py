import sys
sys.path.append('result')
import configparser
from scipy import ndimage
import numpy as np
import fish_track as ft
try:
    from keras.models import load_model
    use_model = True
except ImportError:
    use_model = False


config = ft.utility.Configure('configure.ini')

indices = config.Kernel.principle_axes
indices = indices.split(',')
indices = [int(i) for i in indices]

cluster_num = config.Kernel.cluster_number

try:
    shapes = np.load('./fish_shape_collection.npy')
except FileNotFoundError:
    shapes = np.load('result/fish_shape_collection.npy')

if use_model:
    fail_mark = float(config.Kernel.fail_mark)
    model = load_model('shape_model.h5')
    shapes = ft.utility.validate(shapes, model, fail_mark)

kernels = ft.kernel.get_kernels(shapes, indices, cluster_num, sigma=0)
np.save('shape_kernels', kernels)
