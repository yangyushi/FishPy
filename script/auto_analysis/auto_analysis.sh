#!/usr/bin/env bash
source configure.sh

if [ ! $1 ]; then
    echo "Please specify trajectory file"
    exit 1
fi

traj_path=$1
traj_name=${traj_path##*/}  # /usr/local/bin/py.cpp -> py.cpp
traj_name=${traj_name%%.*}  # py.cpp -> py
root="analysis-${traj_name}"

if [ ! -d "${root}" ]; then
    mkdir "${root}"
fi

cp script/get_tank.py ${root}
cd ${root}
$PY get_tank.py ../input/tank_centres.json ../input
cd ..

if [ $location -ge 1 ]; then
    cp script/get_location.py ${root}
    cd ${root}
    if [ ! -d "spatial_distribution" ]; then
        mkdir spatial_distribution
    fi
    $PY get_location.py ../${traj_path} spatial_distribution
    cd ..
fi

if [ $speed -ge 1 ]; then
    cp script/get_velocity.py ${root}
    cd ${root}
    if [ ! -d "velocity_distribution" ]; then
        mkdir velocity_distribution
    fi
    $PY get_velocity.py ../${traj_path} velocity_distribution
    cd ..
fi

if [ $nn_analysis -ge 1 ]; then
    cp script/get_nn.py ${root}
    cd ${root}
    if [ ! -d "nn_analysis" ]; then
        mkdir nn_analysis
    fi
    $PY get_nn.py ../${traj_path} nn_analysis $nn_ignore_vertices $nn_maximum_length
    cd ..
fi

if [ $dynamical_order -ge 1 ]; then
    cp script/get_dynamic.py ${root}
    cd ${root}
    if [ ! -d "dynamical_analysis" ]; then
        mkdir dynamical_analysis
    fi
    $PY get_dynamic.py ../${traj_path} dynamical_analysis $fps $body_length\
        $frame_start $frame_stop $bin_start $bin_stop $bin_number
fi
