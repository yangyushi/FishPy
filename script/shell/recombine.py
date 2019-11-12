import glob
import pickle
import numpy as np

f = open('locations_3d.pkl', 'wb')
frames = glob.glob(r'locations_3d/frame_*.npy')
frames.sort()
for frame in frames:
    pickle.dump(np.load(frame), f)
f.close()
