import os
import pickle
import numpy as np
import configparser
import matplotlib.pyplot as plt


import fish_track as ft

conf = configparser.ConfigParser(allow_no_value=True)
conf.read('configure.ini')

folder = conf['file']['video_folder']
refine_otol = float(conf['locate']['otol'])
angle_number = int(conf['locate']['angle_number'])
frame_start = int(conf['locate']['frame_start'])
frame_end = int(conf['locate']['frame_end'])
img_threshold = float(conf['locate']['intensity_threshold'])
size_min = int(conf['locate']['size_min'])
size_max = int(conf['locate']['size_max'])
want_plot = bool(int(conf['locate']['want_plot']))
line_length = int(conf['locate']['line_length'])
dpi = 300


for name in conf['rename']:

    if f'features-{name}.pkl' in os.listdir('.'):
        continue

    vid_name = os.path.join(folder, name + '-fg.avi')
    if want_plot:
        vid_raw = ft.read.iter_video(os.path.join(folder, name + '.mp4'))
    images = ft.read.iter_video(vid_name)
    kernels = np.load(f'{name}-shape-kernels.npy')

    angles = np.linspace(0, 180, angle_number)

    f_out = open(f'features-{name}.pkl', 'wb')

    oishi_kernels = ft.get_oishi_kernels(kernels, rot_num=angle_number)

    for frame, image in enumerate(images):
        if frame < frame_start:
            continue
        elif (frame_end > 0) and (frame > frame_end):
            break

        maxima = ft.oishi.get_oishi_features(
            image, oishi_kernels, img_threshold, size_min,
        )

        print(
            f'frame {frame: ^10} feature number {maxima.shape[1]: ^10}',
            end=' -refine-> '
        )

        maxima = ft.refine_oishi_features(
            features=maxima,
            rot_num=angle_number,
            dist_threshold=size_max / 2,
            orient_threshold=refine_otol,
            likelihood_threshold=0,
            intensity_threshold=0
        )

        print(maxima.shape[1])

        pickle.dump(maxima, f_out)

        x, y, o, s, b, p = maxima

        if want_plot:
            raw = next(vid_raw)
            plt.figure(figsize=(raw.shape[1]/dpi, raw.shape[0]/dpi), dpi=dpi)
            length = line_length

            for i, m in enumerate(maxima.T):
                angle = angles[int(o[i])] / 180 * np.pi
                base = m[:2].astype(np.float64)

                plt.plot(
                    [base[0] - length/2 * np.sin(angle), base[0] + length/2 * np.sin(angle)],
                    [base[1] - length/2 * np.cos(angle), base[1] + length/2 * np.cos(angle)],
                    color='tomato', linewidth=0.2,
                )

            plt.imshow(raw, cmap='gray')
            plt.scatter(x, y, color='w', edgecolor='tomato', marker='o', linewidth=0.2, s=3, fc='none')
            plt.xlim(0, raw.shape[1])
            plt.ylim(raw.shape[0], 0)
            plt.gcf().set_frameon(False)
            plt.axis('off')
            plt.gcf().axes[0].get_xaxis().set_visible(False)
            plt.gcf().axes[0].get_yaxis().set_visible(False)
            plt.savefig(f'{name}-oishi-locate-frame-{frame:04}.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()

    f_out.close()
