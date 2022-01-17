#!/usr/bin/env bash
export prefix="/usr/local"
export PY="python3"
export CPLEX_ROOT="/Applications/CPLEX_Studio1210"
export CC="clang++"

# find CPLEX related directories
Machine=`uname -m | sed "s/_/-/g"`
Platform=`uname | tr '[:upper:]' '[:lower:]'`
if [[ $Platform == "darwin" ]]; then
    Platform="osx"
fi
export CPLEX_ARCH=${Machine}_${Platform}

echo "building the voro++ module"
cd extern/voro++
make install
cd ../..

echo "building the fish_corr module"
cd lib/fish_corr
make clean
make all
cd ../..

echo "building the fish_track module"
cd lib/fish_track
make clean
make all
cd ../..

echo "building the fish_3d module"
cd lib/fish_3d
make clean
make all
cd ../..
