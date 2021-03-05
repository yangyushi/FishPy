[![DOI](https://zenodo.org/badge/179326383.svg)](https://zenodo.org/badge/latestdoi/179326383) [![time tracker](https://wakatime.com/badge/github/yangyushi/FishPy.svg)](https://wakatime.com/badge/github/yangyushi/FishPy)

# FishPy - 3D reconstruction

## What is this

This is the code I wrote for my PhD project in university of Bristol.

This is not meant to be used *easily* as I am the only contributer. I will work on making it more user-friendly but this is what I could do for now. Please [write me an email](mailto:yy17363@bristol.ac.uk?subject=Chatting%20about%20FishPy%20) if you are interested in anything related 3D tracking.

## How to use the code

### Install the package

First of all, add the **full path** of folder `FishPy` to the `PYTHONPATH`.

If you know what you are doing, you can also paste the folder to the `site-package` of the pyhton that you are using.

## Cameras

To perform 3D position calculation, we need to know (almost) everything about our cameras. I use my own class `camera` to store the camera parameters (intrinsic/extrinsic parameters & distortion coefficients). Also the class comes with functions from `opencv` that distort/undistort 3D coordinates.

I use the same cameras all the time, so their intrinsic parameters and distortion coefficients were re-used. If you want to use the code please do the following,

1. Use your camera and take many pictures of the chessboard pattern.
2. Use function `Camera.calibrate_int`, read these images and relevant parameters would be updated automatically.
3. Export the camera instances using `pickle`


## Calibration


We need to use *our camera* and take some chessboard images to calculate the camera parameters. A typical image for me [looks like this](../images/calibrate.png), with 4 letters "x", "1", "2", "3" on each corners.

When opencv detect the corners, the *order* matters. You need to provide the order, something like `"x123"`, for everal images to get the extrinsic parameters correctly, because the world origin is supposed to be at "x".

The order can be obtained by the [GUI application](../fish_gui/readme.md) `fish_gui.calib_order`. To get the order, open the application by typing `python3 -m fish_gui.calib_order` and drag the image to the window. Orange lines should appear that connects all the corners on the chessboard. The starting point is represented by a white scatter. If the white scatter falls on "x" then the order is "x123", otherwise the order is "321x". If you wait for quit some time after dropping the image, but eventually nothing appear, you need to take a better image of the chessboard. A good image for corner detection [looks like this](../images/calibrate.png) while a bad image for corner detection [looks like this](../images/calibrate_bad.png).

