"""
support only python3 because of __dict__.items()
it's 2019 now come on
"""
import configparser
import numpy as np
from scipy import ndimage


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
            elif value[1:].isdigit():  # e.g. -12
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
            if section is not "DEFAULT":
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
