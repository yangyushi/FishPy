#import matplotlib
#matplotlib.use('Agg')
import sys
sys.path.append('result')
import numpy as np
from scipy import ndimage 
import matplotlib.pyplot as plt
import fish_track as ft
import configparser
import pickle

config = ft.utility.Configure('configure.ini')

data_path = config.Data.path
data_type = config.Data.type

if config.Process.gaussian_sigma != 0:
    def denoise(x): return ndimage.gaussian_filter(x, blur)
else:
    def denoise(x): return x

if config.Process.normalise == 'std':
    def normalise(x): return x / x.std()
elif config.Process.normalise == 'max':
    def normalise(x): return x / x.max()
elif config.Process.normalise == 'None':
    def normalise(x): return x


roi = config.Process.roi
x0, y0, size_x, size_y = [int(x) for x in roi.split(', ')]
roi = (slice(y0, y0 + size_y, None), slice(x0, x0 + size_x))

refine_otol = config.Refine.otol

blur  = config.Process.gaussian_sigma
angle_number = config.Locate.orientation_number

frame_start = config.Locate.frame_start
frame_end = config.Locate.frame_end
cc_threshold = config.Locate.cc_threshold
img_threshold = config.Fish.threshold

if data_type == 'video':
    images = ft.read.iter_video(data_path)
elif data_type == 'images':
    images = ft.read.iter_image_sequence(data_path)
else:
    raise TypeError("Wrong data type", data_type)

try:
    background = np.load('background.npy')
except FileNotFoundError:
    background = np.load('result/background.npy')

try:
    kernels = np.load('shape_kernels.npy')
except FileNotFoundError:
    kernels = np.load('result/shape_kernels.npy')

# the degree 180 is not included, it should be cuvered by another "upside-down" shape
angles = np.linspace(0, 180, angle_number)

f_out = open('features.pkl', 'wb')

dpi = 150

for frame, image in enumerate(images):

    if frame < frame_start:
        continue
    elif frame > frame_end:
        break
    else:
        pass

    fg = background - normalise(denoise(image))
    fg -= fg.min()
    fg = fg[roi]

    cross_correlation = ft.oishi.get_cross_correlation_nd(
            fg, angles, kernels
            )

    maxima = ft.oishi.oishi_locate(
            fg, cross_correlation, config.Fish.size_min,
            cc_threshold, img_threshold
            )

    maxima = ft.oishi.oishi_refine(maxima, angles, config.Fish.size_max, otol=refine_otol)


    pickle.dump(maxima, f_out)

    o, r, x, y, p = maxima

    if config.Plot.want_plot == 'True':
        plt.figure(figsize=(fg.shape[1]/dpi, fg.shape[0]/dpi), dpi=dpi)
        length = config.Plot.line_length
        for i, m in enumerate(maxima.T):
            angle = angles[o[i]] / 180 * np.pi
            base = m[2:].astype(np.float64)
            plt.plot(
                [base[1] - length/2 * np.sin(angle), base[1] + length/2 * np.sin(angle)],
                [base[0] - length/2 * np.cos(angle), base[0] + length/2 * np.cos(angle)],
                color='tomato', linewidth=1,
            )
        plt.imshow(image[roi], cmap='gray')
        plt.scatter(y, x, color='w', edgecolor='tomato', marker='o', linewidth=1, s=12)
        plt.xlim(0, fg.shape[1])
        plt.ylim(fg.shape[0], 0)
        plt.gcf().set_frameon(False)
        plt.axis('off')
        plt.gcf().axes[0].get_xaxis().set_visible(False)
        plt.gcf().axes[0].get_yaxis().set_visible(False)
        plt.savefig(f'oishi_locate_frame_{frame + frame_start:04}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

f_out.close()
