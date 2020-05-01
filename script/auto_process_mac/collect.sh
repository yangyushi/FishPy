#!/usr/bin/env bash


succeed_2d=1
for i in {1..3}; do
    feature="track_2d/features_2d-cam_$i.pkl"
    shape="track_2d/shapes_2d-cam_$i.npy"
    if [[ ! -e $feature ]] || [[ ! -e $shape ]]; then
        succeed_2d=0
    fi
done

succeed_3d=1
for i in {1..3}; do
    cam="track_3d/cam_$i.pkl"
    if [[ ! -e $cam ]]; then
        succeed_3d=0
    fi
done
if [[ ! -e "track_3d/locations_3d.pkl" ]]; then
    succeed_3d=0
fi

if [[ $succeed_2d -ne 1 ]]; then
    echo "2D Tracking Failed"
    exit 1
fi
if [[ $succeed_3d -ne 1 ]]; then
    echo "3D Tracking Failed"
    exit 1
fi

if [[ -d "results" ]]; then
    rm -rf results
fi
mkdir results

python3 script/collect.py results

if [[ $1 == "clear" ]]; then
    source configure.sh
    rm -rf $video_folder
fi
