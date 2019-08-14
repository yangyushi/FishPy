# FishPy

## What is this

This is the code I wrote for my PhD project in university of Bristol.

This is not meant to be used *easily*. I will work on making it more user-friendly but this is what I could do for now.


## How to use the code

1. Copy the scripts under `script` folder
2. Read the `Makefile`, konwing what's going on.
3. Modify the configuration files (`*.ini`)
4. Use `make` in terminal, getting results

## Use the code (even with a GUI)

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

After the path to the directory is added to the `$PYTHONPATH$`
