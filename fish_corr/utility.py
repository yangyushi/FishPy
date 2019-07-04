#!/usr/bin/env python3
import numpy as np


def auto_corr(var, dt=1):
    bar = var - np.mean(var)
    stop = len(var)
    c = np.mean(np.sum(bar[dt:stop] * bar[:stop-dt]))
    c0 = np.mean(np.sum(bar * bar))
    return c/c0


def get_acf(var):
    """
    Calculate the auto-correlation function for a 1D variable
    """
    size = len(var)
    result = np.zeros(size - 1, dtype=np.float64)
    for dt in range(0, size - 1):
        result[dt] = auto_corr(var, dt)
    return result

