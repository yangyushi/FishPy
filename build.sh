#!/usr/bin/env bash
export prefix=/usr/local
export PY=python3
export CPLEX_ROOT="/Applications/CPLEX_Studio1210"
export CC="clang++"

# find CPLEX related directories
Machine=`uname -m | sed "s/_/-/g"`
Platform=`uname | tr '[:upper:]' '[:lower:]'`
if [[ $Platform == "darwin" ]]; then
    Platform="osx"
fi
export CPLEX_ARCH=${Machine}_${Platform}


cd fish_3d
make all
cd ../fish_corr
make all
cd ../fish_track
make all
cd ..
