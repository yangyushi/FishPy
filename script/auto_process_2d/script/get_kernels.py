import os
import re
import numpy as np
import configparser

if 'shape_kernels.npy' in os.listdir('.'):
    exit(0)

import fish_track as ft
try:
    from tensorflow import keras
    use_model = True
except ImportError:
    use_model = False


conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

indices = conf['locate']['principle_axes']
indices = re.split(r',\s*', indices)
indices = [int(i) for i in indices]

cluster_num = int(conf['locate']['cluster_number'])
shapes = np.load('fish_shape_collection.npy')

if use_model:
    fail_mark = float(conf['locate']['fail_mark'])
    model = keras.models.load_model('script/shape_model.h5')
    shapes = ft.utility.validate(shapes, model, fail_mark)

kernels = ft.kernel.get_kernels(shapes, indices, cluster_num, sigma=0)
np.save('shape_kernels', kernels)
