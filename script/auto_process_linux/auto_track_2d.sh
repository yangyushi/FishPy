#!/bin/bash
# request resources:
#PBS -N fish_track_2d
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00

if [ $PBS_O_WORKDIR ]; then
    cd $PBS_O_WORKDIR
fi

# retrieving parameters
source configure.sh

# preprocessing videos
./script/rename_vid.sh $video_folder
python3 script/preprocess.py "${video_folder}/cam-1.mp4" $background_rolling_length $blur $local $binary_open_size
python3 script/preprocess.py "${video_folder}/cam-2.mp4" $background_rolling_length $blur $local $binary_open_size
python3 script/preprocess.py "${video_folder}/cam-3.mp4" $background_rolling_length $blur $local $binary_open_size

# create folders for 2D tracking
if [ ! -d "track_2d" ]; then
    mkdir track_2d
fi

if [ ! -d "track_2d/cam-1" ]; then
    mkdir track_2d/cam-1
fi

if [ ! -d "track_2d/cam-2" ]; then
    mkdir track_2d/cam-2
fi

if [ ! -d "track_2d/cam-3" ]; then
    mkdir track_2d/cam-3
fi

# prepare for 2d tracking (camera 1)
cp $script_folder/track_2d/* track_2d/cam-1
# fill the location of the video file
video_folder_escaped=$(echo $video_folder | sed -e 's~/[]./[]~\&~g')
sed -i'' "s~VIDEOPATH~\.\./\.\./$video_folder_escaped/cam-1-fg\.avi~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DFRAMESTART~$track_2d_frame_start~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DFRAMEEND~$track_2d_frame_end~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DFRAMEEND~$track_2d_frame_end~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DSIZEMAX~$track_2d_size_max~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DSIZEMIN~$track_2d_size_min~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DINTENSITYTHRESHOLD~$track_2d_intensity_threshold~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DWANTPLOT~$track_2d_want_plot~" track_2d/cam-1/configure.ini
sed -i'' "s~TRACK2DORIENTATIONNUMBER~$track_2d_orientation_number~" track_2d/cam-1/configure.ini

if [ -e "$cam_1_kernels" ]; then
    cp $cam_1_kernels track_2d/cam-1
    has_kernel_cam_1=1
else
    has_kernel_cam_1=0
fi
if [ $measure_roi -gt 0 ]; then
    cd track_2d/cam-1
    make roi
    cd -
fi

# prepare for 2d tracking (camera 2)
cp $script_folder/track_2d/* track_2d/cam-2
# fill the location of the video file
sed -i'' "s~VIDEOPATH~\.\./\.\./$video_folder/cam-2-fg\.avi~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DFRAMESTART~$track_2d_frame_start~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DFRAMEEND~$track_2d_frame_end~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DFRAMEEND~$track_2d_frame_end~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DSIZEMAX~$track_2d_size_max~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DSIZEMIN~$track_2d_size_min~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DINTENSITYTHRESHOLD~$track_2d_intensity_threshold~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DWANTPLOT~$track_2d_want_plot~" track_2d/cam-2/configure.ini
sed -i'' "s~TRACK2DORIENTATIONNUMBER~$track_2d_orientation_number~" track_2d/cam-2/configure.ini

if [ -e "$cam_2_kernels" ]; then
    cp $cam_2_kernels track_2d/cam-2
    has_kernel_cam_2=1
else
    has_kernel_cam_2=0
fi
if [ $measure_roi -gt 0 ]; then
    cd track_2d/cam-2
    make roi
    cd -
fi

# prepare for 2d tracking (camera 3)
cp $script_folder/track_2d/* track_2d/cam-3
# fill the location of the video file
sed -i'' "s~VIDEOPATH~\.\./\.\./$video_folder/cam-3-fg\.avi~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DFRAMESTART~$track_2d_frame_start~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DFRAMEEND~$track_2d_frame_end~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DFRAMEEND~$track_2d_frame_end~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DSIZEMAX~$track_2d_size_max~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DSIZEMIN~$track_2d_size_min~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DINTENSITYTHRESHOLD~$track_2d_intensity_threshold~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DWANTPLOT~$track_2d_want_plot~" track_2d/cam-3/configure.ini
sed -i'' "s~TRACK2DORIENTATIONNUMBER~$track_2d_orientation_number~" track_2d/cam-3/configure.ini


if [ -e "$cam_3_kernels" ]; then
    cp $cam_3_kernels track_2d/cam-3
    has_kernel_cam_3=1
else
    has_kernel_cam_3=0
fi
if [ $measure_roi -gt 0 ]; then
    cd track_2d/cam-3
    make roi
    cd -
fi

# 2D Tracking
cd track_2d/cam-1
if [ $has_kernel_cam_1 -gt 0 ]; then
    make feature
else
    make shape
    make kernel
    make feature
fi
make sort
cd -

cd track_2d/cam-2
if [ $has_kernel_cam_2 -gt 0 ]; then
    make feature
else
    make shape
    make kernel
    make feature
fi
make sort
cd -

cd track_2d/cam-3
if [ $has_kernel_cam_3 -gt 0 ]; then
    make feature
else
    make shape
    make kernel
    make feature
fi
make sort
cd -

cp track_2d/cam-1/result/features.pkl track_2d/features_2d-cam_1.pkl
cp track_2d/cam-1/result/shape_kernels.npy track_2d/shapes_2d-cam_1.npy

cp track_2d/cam-2/result/features.pkl track_2d/features_2d-cam_2.pkl
cp track_2d/cam-2/result/shape_kernels.npy track_2d/shapes_2d-cam_2.npy

cp track_2d/cam-3/result/features.pkl track_2d/features_2d-cam_3.pkl
cp track_2d/cam-3/result/shape_kernels.npy track_2d/shapes_2d-cam_3.npy
