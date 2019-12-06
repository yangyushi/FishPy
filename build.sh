export prefix=/usr/local
export PY=python3

cd fish_3d
make all
cd ../fish_corr
make all
cd ../fish_track
make all
cd ..
