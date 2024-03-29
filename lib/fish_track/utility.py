"""
support only python3 because of __dict__.items()
it's 2019 now come on
"""
import configparser
import numpy as np
from scipy import ndimage
from scipy.signal import correlate2d


class SubProperty:
    def __init__(self, dictionary):
        """
        data type is automatically converted here
        if can be int, be int, then
        if can be float, be float, otherwise be str
        """
        assert isinstance(dictionary, dict), "a dict is needed"
        for key, value in dictionary.items():
            if value.isdigit():  # e.g. 12
                setattr(self, key, int(value))
            elif value[1:].isdigit() and value[0] == '-':  # e.g. -12
                    setattr(self, key, int(value))
            else:
                try:  # e.g. 12.3
                    setattr(self, key, float(value))
                except ValueError:
                    setattr(self, key, value)

    def __iter__(self):
        for attr, val in self.__dict__.items():
            yield attr, val


class Configure:
    def __init__(self, config_file):
        """
        convert python configparser into a class, just make it more convenient to use
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        for section in config:
            if section != "DEFAULT":
                dict_property = dict(config[section])
                setattr(self, section, SubProperty(dict_property))

    def __iter__(self):
        for attr, val in self.__dict__.items():
            yield attr, val

    def write(self, config_file):
        config = configparser.ConfigParser()
        for section_name, section in self:
            config[section_name] = {}
            for item_name, item_value in section:
                config[section_name][item_name] = str(item_value)
        with open(config_file, 'w') as f:
            config.write(f)


def join_pairs(pairs):
    if len(pairs) == 0:
        return []
    max_val = np.max(np.hstack(pairs)) + 1
    canvas = np.zeros((max_val, max_val), dtype=int)
    p = np.array(pairs)
    canvas[tuple(p.T)] = 1
    labels, _ = ndimage.label(canvas)
    joined_pairs = []
    for val in set(labels[labels > 0]):
        joined_pair = np.unique(np.vstack(np.where(labels == val)))
        joined_pairs.append(joined_pair)
    return joined_pairs


def validate(images, model, fail_mark=0.25, shape=(40, 40)):
    """
    validate shapes with trained keras model
    good shape: probability of being good > fail_mark
    """
    size_x, size_y = images[0].shape
    zoom = (1, shape[0] / size_x, shape[1] / size_y)
    images = ndimage.zoom(images, zoom)
    norm_factor = images.max(1).max(1)
    norm_factor = np.expand_dims(norm_factor, -1)
    norm_factor = np.expand_dims(norm_factor, -1)
    normed = images / norm_factor
    normed = np.expand_dims(normed, -1)  # (n, x, y) -> (n, x, y, 1)
    score = model.predict(normed).ravel()
    good_indices = np.where(score > fail_mark)
    return images[good_indices]


def draw_2d(radius: int):
    length = 2 * radius + 1
    canvas = np.zeros((length, length))
    for idx in range(length):
        for idy in range(length):
            if (idx - radius) ** 2 + (idy - radius) ** 2 <= radius**2:
                canvas[idx, idy] = 1
    return canvas


def detect_circle(image: np.array, size=50, upper_threshold=0.5):
    """
    Detect the central circle in the image
    The foreground is dark
    :param image: 2d image as numpy array
    :return: the (x, y) coordinates of the centre
    """
    radii = np.linspace(size/3, size/2, 10, dtype=int)
    zoom = min(image.shape) // size
    scale = float(min(image.shape)) / size
    small = image[::zoom, ::zoom].astype(np.float64)

    to_detect = small.max() - small.astype(float)
    to_detect[small==0] = 0
    max_val = np.max(to_detect) * upper_threshold
    to_detect[to_detect > max_val] = max_val
    to_detect[to_detect < np.mean(to_detect)] = np.mean(to_detect)

    positions = []
    similarities = []

    for r in radii:
        sim = draw_2d(r)
        corr = correlate2d(
            to_detect - np.min(to_detect),
            sim - sim.mean(),
            mode='same'
        )
        try:
            pos = np.unravel_index(np.argmax(corr), corr.shape)
            similarity = corr[pos] / sim.sum()
        except ValueError:
            pos = np.array((0, 0))
            similarity = 0
        positions.append(pos)
        similarities.append(similarity)

    idx = np.argmax(similarities)
    centre = np.array(positions[idx])
    radius = radii[idx]
    return centre[::-1] * scale, radius * scale, to_detect
