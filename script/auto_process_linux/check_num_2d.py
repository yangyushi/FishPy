import pickle
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


for i in (1, 2, 3):
    numbers = []

    f = open(f'track_2d/features_2d-cam_{i}.pkl', 'rb')

    while True:
        try:
            features = pickle.load(f)
            n = features.shape[1]
            numbers.append(n)
        except EOFError:
            break

    f.close()
    color = cm.rainbow(i / 4.0)
    plt.plot(
        numbers, label=f'cam_{i}', color=color,
    )

plt.legend()
plt.show()
