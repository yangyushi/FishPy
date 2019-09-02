    #!/usr/bin/env python3
import pickle
import fish_3d as f3
from glob import glob

cam_files = glob('cam*.pkl')
cam_files.sort()

cameras = {}

for i, fn in enumerate(cam_files):
    with open(fn, 'rb') as f:
        cam = pickle.load(f)
    cameras.update({f'camera_{i+1}': cam})

with open('cameras.pkl', 'wb') as f:
    pickle.dump(cameras, f)
