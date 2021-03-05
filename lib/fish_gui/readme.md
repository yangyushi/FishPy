[![DOI](https://zenodo.org/badge/179326383.svg)](https://zenodo.org/badge/latestdoi/179326383) [![time tracker](https://wakatime.com/badge/github/yangyushi/FishPy.svg)](https://wakatime.com/badge/github/yangyushi/FishPy)

# FishPy - GUI application

## What is this

This is the code I wrote for my PhD project in university of Bristol.

This is not meant to be used *easily* as I am the only contributer. I will work on making it more user-friendly but this is what I could do for now. Please [write me an email](mailto:yy17363@bristol.ac.uk?subject=Chatting%20about%20FishPy%20) if you are interested in anything related 3D tracking.

I wrote some helper GUI applications in this module

## How to use the code

### Install the package

First of all, add the **full path** of folder `FishPy` to the `PYTHONPATH`.

If you know what you are doing, you can also paste the folder to the `site-package` of the pyhton that you are using.

### Use the GUI applications

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
