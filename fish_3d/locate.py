import numpy as np
from scipy import ndimage 

def mini_colloids(image, k, layers, min_intensity, size, remove_noise):
    if remove_noise:
        new_image = ndimage.gaussian_filter(image, remove_noise)
    else:
        new_image = image
    dogs = np.zeros([
        layers,
        image.shape[0],
        image.shape[1]
    ])
    tmp = (new_image - new_image.min()) / (new_image.max() - new_image.min())
    dogs = np.diff([
            ndimage.gaussian_filter(tmp, s) for s in k * (2 ** (1.0/layers)) ** np.arange(layers + 3)
            ], axis=0)
    cats = dogs.max() - dogs
    cats = cats / cats.max()
    maxima = (ndimage.grey_dilation(cats, [layers, size, size]) == cats) * (cats > min_intensity * cats.max())
    maxima = np.array(np.where(maxima > 0))
    return maxima[1:]  # (dimension x number)
