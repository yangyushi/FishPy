[![DOI](https://zenodo.org/badge/179326383.svg)](https://zenodo.org/badge/latestdoi/179326383) [![time tracker](https://wakatime.com/badge/github/yangyushi/FishPy.svg)](https://wakatime.com/badge/github/yangyushi/FishPy)

# FishPy - Automated analysis

## What is this

This is the code I wrote for my PhD project in university of Bristol.

This is not meant to be used *easily* as I am the only contributer. I will work on making it more user-friendly but this is what I could do for now. Please [write me an email](mailto:yy17363@bristol.ac.uk?subject=Chatting%20about%20FishPy%20) if you are interested in anything related 3D tracking.

## How to use the code

### Install the package

First of all, add the **full path** of folder `FishPy` to the `PYTHONPATH`.

If you know what you are doing, you can also paste the folder to the `site-package` of the pyhton that you are using.

### Run the script

1. Get a list of [`Trajectory`](../../fish_track/linking.py) instances and dump the list use `pickle`.
2. Modify the `configure.sh` file accordingly.
3. Type `./auto_analysis.sh _trajectories_file_`
4. Get results
