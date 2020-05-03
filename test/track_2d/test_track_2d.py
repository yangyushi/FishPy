from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import fish_track as ft


shape_kernels = np.load('shape_kernels.npy')
img = np.array(Image.open('fish-50.png').convert('L'))


def test_oishi_feature():
    oishi_kernels = ft.get_oishi_kernels(shape_kernels, rot_num=36)
    maxima = ft.oishi.get_oishi_features(
        img, oishi_kernels, threshold=0.5, local_size=5
    )
    maxima = ft.refine_oishi_features(
        features=maxima,
        rot_num=36,
        dist_threshold=15,
        orient_threshold=30,
        likelihood_threshold=2.0,
        intensity_threshold=0.5
    )


if __name__ == "__main__":
    test_oishi_feature()
