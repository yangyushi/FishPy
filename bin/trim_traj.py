import pickle
import numpy as np
import matplotlib.pyplot as plt


filename = "trajectories-video-5-raw.pkl"

with open(filename, 'rb') as f:
    trajs = pickle.load(f)


X, Y, Z = [], [], []

for t in trajs:
    x, y, z = t['position'].mean(axis=0)
    X.append(x)
    Y.append(y)
    Z.append(z)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

X -= np.mean(X, 0)
Y -= np.mean(Y, 0)

R = np.sqrt(X**2 + Y**2)


hist, binx, biny = np.histogram2d(R, Z, bins=50)
plt.imshow(hist.T)
ticks = np.linspace(0, 50, 5, endpoint=True, dtype=int)
plt.gca().set_xticks(ticks)
plt.gca().set_yticks(ticks)
plt.gca().set_xticklabels([f'{binx[i]:.2f}' for i in ticks])
plt.gca().set_yticklabels([f'{biny[i]:.2f}' for i in ticks])
plt.ylim(0, 49)
plt.show()

new_fn = filename[:-8] + '.pkl'

new_trajs = [t for i, t in enumerate(trajs) if (R[i] < 600) or (Z[i] > -320)]
print(len(new_trajs))
new_trajs = [t for i, t in enumerate(new_trajs) if len(t['time']) > 5]
print(len(new_trajs))

with open(new_fn, 'wb') as f:
    pickle.dump(new_trajs, f)


X, Y, Z = [], [], []

for t in new_trajs:
    x, y, z = t['position'].mean(axis=0)
    X.append(x)
    Y.append(y)
    Z.append(z)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

X -= np.mean(X, 0)
Y -= np.mean(Y, 0)

R = np.sqrt(X**2 + Y**2)

hist, binx, biny = np.histogram2d(R, Z, bins=50)
plt.imshow(hist.T)
ticks = np.linspace(0, 50, 5, endpoint=True, dtype=int)
plt.gca().set_xticks(ticks)
plt.gca().set_yticks(ticks)
plt.gca().set_xticklabels([f'{binx[i]:.2f}' for i in ticks])
plt.gca().set_yticklabels([f'{biny[i]:.2f}' for i in ticks])
plt.ylim(0, 49)
plt.show()
