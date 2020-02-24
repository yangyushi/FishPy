[![DOI](https://zenodo.org/badge/179326383.svg)](https://zenodo.org/badge/latestdoi/179326383) [![time tracker](https://wakatime.com/badge/github/yangyushi/FishPy.svg)](https://wakatime.com/badge/github/yangyushi/FishPy)

# FishPy

## What is this

This is the code I wrote for my PhD project in university of Bristol.

This is not meant to be used *easily*. I will work on making it more user-friendly but this is what I could do for now.


## How to use the code

### Install the package

First of all, add the **full path** of folder `FishPy` to the `PYTHONPATH`.

If you know what you are doing, you can also paste the folder to the `site-package` of the pyhton that you are using.

### Setting up the project

To do the 2D and 3D tracking, I would copy the folder `script/auto_process_[mac|linux]` to the folder that I have my data. The sturecture of the folders is like this

```
project_folder
├── auto_process
├── calib-ext
│   ├── calib-orders.json
│   ├── cam_1-1.tiff
│   ├── cam_1-2.tiff
│   ├── cam_1-3.tiff
│   ├── cam_2-1.tiff
│   ├── cam_2-2.tiff
│   ├── cam_2-3.tiff
│   ├── cam_3-1.tiff
│   ├── cam_3-2.tiff
│   ├── cam_3-3.tiff
│   └── tags
└── movie
    ├── cam-1.mp4
    ├── cam-2.mp4
    └── cam-3.mp4
```

And the `cam_#-#.tiff` file is typically the images taking to get the extrinsic (location & rotation) of the cameras. Yes the naming style is FIXED very unfortunatly, sorry! The images need to be chessboard looks like this, with 4 letters "x", "1", "2", "3" on each corners.

The content inside file `calib-orders.json` is typically like this

```
{
  "cam_1": ["x123", "x123", "x123"],
  "cam_2": ["x123", "x123", "x123"],
  "cam_3": ["321x", "321x", "321x"]
}
```

The *orders*, something like `"x123"`, is obtained by the GUI application `fish_gui.calib_order`. To get the order, open the application by typing `python3 -m fish_gui.calib_order` and drag the image to the window. Orange lines should appear that connects all the corners on the chessboard. The starting point is represented by a white scatter. If the white scatter falls on "x" then the order is "x123", otherwise the order is "321x". If you wait for quit some time after dropping the image, but eventually nothing appear, you need to take a better image of the chessboard. A good image for corner detection looks like this.

### Run the code

Next we `cd` to the `auto_process_[mac|linux]` folder. Change the `configure.sh` for tracking parameters. Make sure correct camera model files in the `database` folder.

For 2D tracking, tpye `./auto_track_2d.sh`.

For 3D tracking, tpye `./auto_track_3d.sh`.

For linking positions into trajectories, type `./auto_link.sh`.

After everything, you should be able to get results! You can type `python3 see_trajs.py show` to see the trajectories in 3D. Good luck!


## Cameras

I use my own class `camera` to store the camera parameters (intrinsic/extrinsic parameters & distortion coefficients). Also the class comes with functions from `opencv` that distort/undistort 3D coordinates.

I use the same cameras all the time, so their intrinsic parameters and distortion coefficients were re-used. If you want to use the code please do the following,

1. Use your camera and take many pictures of the chessboard pattern.
2. Use function `Camera.calibrate_int`, read these images and relevant parameters would be updated automatically.
3. Export the camera instances using 

## Use the GUI applications

There are three pieces of code that runs interactively.

1. `epipolar`: read two calibrated cameras and shows the epipolar relationship.
2. `shape_gallery`: the interactive fish shape inspector I built for labelling bad fish shape
3. `calib_order`: a quick tool to check the order of the calibration chessboard

The way to use these app is

```sh
python3 -m fish_gui.epipolar
python3 -m fish_gui.shape_gallery
python3 -m fish_gui.calib_order
```
