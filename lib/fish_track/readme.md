[![DOI](https://zenodo.org/badge/179326383.svg)](https://zenodo.org/badge/latestdoi/179326383) [![time tracker](https://wakatime.com/badge/github/yangyushi/FishPy.svg)](https://wakatime.com/badge/github/yangyushi/FishPy)

# Fishpy - 2D Tracking

## What is this

This package contains code for 2D tracking & ND linking.

- 2D Tracking
    - `shape.py`: segment image and extract fish shapes.
    - `kernel.py`: use PCA to calculate representative fish shapes, noted as the *kernel*.
    - `oishi.py`: use the kernels to find oishi features. Each feature contains following information
        1. x coordinate
        2. y coordinate
        3. orientation
        4. shape
        5. brightness
        6. likelihood
    - `utility.py`: contains several helper funcitions
- Linking
    - `linking.py`: link n-dimentional positions into trajectories, and manage trajectories.
    - `nrook.cpp`: helper module for `linking.py`, binded with [`pybind11`](https://github.com/pybind/pybind11)


## How to use the code

First of all, add the **full path** of folder `FishPy` to the `PYTHONPATH`.

If you know what you are doing, you can also paste the folder to the `site-package` of the pyhton that you are using.

### 2D Tracking

A *standard template* for 2D tracking is availabe in `FishPy/script/track`. To perform a 2D tracking, copy the folder to a preferred place (where the tracking result will be held), and follow this,

1. Edit `configure.ini` *accordingly*.
2. In the terminal, type `make roi` and manually select the region of interest. Using a smaller ROI reduces the computational time.
3. Use `make shapes` to perform segmentation & extract fish shapes.
4. Use `make kernel` to get kernels
5. Use `make feature` to calculate oishi features


### Linking

A *standard template* for the linking is available in `FishPy/script/auto_process_linux|mac/script/link.py`.
