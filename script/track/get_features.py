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

roi = config.Process.roi
x0, y0, size_x, size_y = [int(x) for x in roi.split(', ')]
roi = (
        slice(y0, y0 + size_y),
        slice(x0, x0 + size_x),
        )

refine_otol = config.Refine.otol

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
    kernels = np.load('shape_kernels.npy')
except FileNotFoundError:
    kernels = np.load('result/shape_kernels.npy')

# the degree 180 is not included, it should be cuvered by another "upside-down" shape
angles = np.linspace(0, 180, angle_number)

f_out = open('features.pkl', 'wb')

dpi = 150

oishi_kernels = ft.get_oishi_kernels(kernels, rot_num=angle_number)
np.save('oishi_kernels', oishi_kernels)

for frame, image in enumerate(images):

    if frame < frame_start:
        continue
    elif frame > frame_end:
        break
    else:
        pass

    fg = image[roi]

    maxima = ft.oishi.get_oishi_features(
            fg, oishi_kernels, img_threshold, config.Fish.size_min,
            )

    print(f'frame {frame: ^10} feature number ', maxima.shape[1], end=' --refine--> ')

    maxima = ft.refine_oishi_features(
        features=maxima,
        rot_num=angle_number,
        dist_threshold=config.Fish.size_max / 2,
        orient_threshold=refine_otol,
        likelihood_threshold=cc_threshold,
        intensity_threshold=0
    )

    print(maxima.shape[1])

    pickle.dump(maxima, f_out)

    x, y, o, s, b, p = maxima

    if config.Plot.want_plot == 'True':
        plt.figure(figsize=(fg.shape[1]/dpi, fg.shape[0]/dpi), dpi=dpi)
        length = config.Plot.line_length

        for i, m in enumerate(maxima.T):
            angle = angles[int(o[i])] / 180 * np.pi
            base = m[:2].astype(np.float64)

            plt.plot(
                [base[0] - length/2 * np.sin(angle), base[0] + length/2 * np.sin(angle)],
                [base[1] - length/2 * np.cos(angle), base[1] + length/2 * np.cos(angle)],
                color='tomato', linewidth=1,

            )

        plt.imshow(image[roi], cmap='gray')
        plt.scatter(x, y, color='w', edgecolor='tomato', marker='o', linewidth=1, s=12)
        plt.xlim(0, fg.shape[1])
        plt.ylim(fg.shape[0], 0)
        plt.gcf().set_frameon(False)
        plt.axis('off')
        plt.gcf().axes[0].get_xaxis().set_visible(False)
        plt.gcf().axes[0].get_yaxis().set_visible(False)
        plt.savefig(f'oishi_locate_frame_{frame:04}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

f_out.close()
