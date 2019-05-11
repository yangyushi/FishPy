import numpy as np
import matplotlib.pyplot as plt
from . import ray_trace

dpi = 150

def plot_reproject(image, pos_3d, camera, filename=None, water_level=0, normal=(0, 0, 1)):
    fig = plt.figure(figsize=(image.shape[1]/dpi, image.shape[0]/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    for point in pos_3d:
        color = (np.random.random(3) + 1) / 2
        xy = ray_trace.reproject_refractive(point, camera)
        ax.scatter(*xy, color='tomato', edgecolor='tomato', facecolor='none', marker='o', lw=2, s=100)

    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.gcf().set_frameon(False)
    plt.gcf().axes[0].get_xaxis().set_visible(False)
    plt.gcf().axes[0].get_yaxis().set_visible(False)
    plt.axis('off')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
