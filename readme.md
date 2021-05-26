[![DOI](https://zenodo.org/badge/179326383.svg)](https://zenodo.org/badge/latestdoi/179326383) [![time tracker](https://wakatime.com/badge/github/yangyushi/FishPy.svg)](https://wakatime.com/badge/github/yangyushi/FishPy)

# FishPy

## What is this

This is the code I wrote for my PhD project in university of Bristol.

This is not meant to be used *easily* as I am the only contributer. I will work on making it more user-friendly but this is what I could do for now. Please [write me an email](mailto:yy17363@bristol.ac.uk?subject=Chatting%20about%20FishPy%20) if you are interested in using this code.


IMHO, the package is *good* in terms of these feaures,

1. Getting 2D coordinates and orientations of a large amount of fish from a 2D video.
2. Reconstructing the 3D locations from *matched* 2D coordinates from multiple views (with water refraction correction).
3. Implementing some useful function to link positions into trajectories.
4. Calculating some correlation functions from the trajectories.
5. Fully automated scripts for above functions, being compatable with HPC.

## How to use the code

### Dependencies

- pybind11
- Eigen3
- Voro++
- cplex
- opencv-python
- numba
- pandas
- pillow
- joblib
- trackpy
- numpy
- scipy
- matplotlib
- tensorflow (optional)

### Install the package

1. Add the full path of folder `FishPy/lib` to the `PYTHONPATH`.
2. Add the full path of folder `FishPy/bin` to the `PATH`.

### Use different modules

Please see the readme file inside each sub-folder for further instructions. Typically,

- [2D Tracking & Linking](fish_track)
- [3D Tracking](fish_3d)
- [Automated tracking](script/auto_process_linux)
- [Automated Analysing](script/auto_analysis)
- [GUI applications](fish_gui)
