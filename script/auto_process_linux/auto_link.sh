#!/bin/bash
# request resources:
#PBS -N fl-
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l mem=32g
#PBS -q himem

if [ $PBS_O_WORKDIR ]; then
    cd $PBS_O_WORKDIR
fi

source configure.sh

# Linking parameters
link_frame_start=0
link_frame_end=0
link_linker="active"
link_range=20
link_dx_max=25
link_dt_max=10
link_blur=1
link_threshold=3
relink_window=500

if [ ! $link_frame_end -gt 0 ]; then
    link_frame_end=$track_3d_frame_end
fi

if [ ! -d "link_3d" ]; then
    mkdir link_3d
fi

python3 script/link.py track_3d/locations_3d.pkl $link_linker\
    $link_frame_start $link_frame_end $link_range\
    $link_dx_max $link_dt_max $link_blur $link_threshold\
    link_3d $relink_window
