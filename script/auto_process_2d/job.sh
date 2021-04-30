capture_err() {
    exit_code=$1
    if [[ $exit_code -ne 0 ]]; then
        exit 1
    fi
}

echo pre-processing videos
python3 script/preprocess.py; capture_err $?

echo calibrating cameras
python3 script/calibrate.py; capture_err $?

echo recitifying video
python3 script/rectify.py; capture_err $?

echo tracking the fish
python3 script/get_shapes.py; capture_err $?
python3 script/get_kernels.py; capture_err $?
python3 script/get_features.py; capture_err $?

echo linking the trajectories
python3 script/link.py; capture_err $?

if [[ ! -d images ]]; then
    mkdir images
fi
mv *.png images
mv *.pdf images

if [[ ! -d data ]]; then
    mkdir data
fi
mv *.npy data
mv *pkl data
mv data/movie.pkl .
