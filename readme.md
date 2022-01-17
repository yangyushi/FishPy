[![DOI](https://zenodo.org/badge/179326383.svg)](https://zenodo.org/badge/latestdoi/179326383) [![time tracker](https://wakatime.com/badge/github/yangyushi/FishPy.svg)](https://wakatime.com/badge/github/yangyushi/FishPy)

# FishPy

## What is this

This is the code I wrote for my PhD project in university of Bristol. The code produced these results:

- https://www.biorxiv.org/content/10.1101/2021.09.01.458490v1

This is not meant to be used *easily* as I am the only contributer.
I will work on making it more user-friendly but this is what I could do for now.

IMHO, the package is *good* in terms of these feaures,

1. Getting 2D coordinates and orientations of a large amount of fish from a 2D video.
2. Reconstructing the 3D locations from *matched* 2D coordinates from multiple views (with water refraction correction).
3. Linking positions into trajectories, for both 2D and 3D data.
4. Calculating spatial & temporal correlation functions.

## Install the package

1. Download the code with command `git clone --recursive https://github.com/yangyushi/FishPy.git`. It is *important* to include the `--recursive` option, so that the `eigen` and `pybind11` will be downloaded.
2. Change the content in `build.sh` accordingly, then execute `./build.sh` to build the python modules. This step is *hard*.
3. [optional] Add the full path of folder `FishPy/lib` to the `PYTHONPATH`, so that the module can be imported in Python.
4. [optional] Add the full path of folder `FishPy/bin` to the `PATH`, to use the scripts in `bin` folder.


## Dependencies

### C/C++

- [cplex](https://pypi.org/project/cplex/) (you need to download the source code and compile. Also you need to install its Python API.)

### Python

- [opencv-python-headless](https://pypi.org/project/opencv-python-headless/)
- numba
- pandas
- pillow
- joblib
- trackpy
- numpy
- scipy
- matplotlib
- tensorflow (optional)
