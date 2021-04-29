echo pre-processing videos
python3 script/preprocess.py

echo calibrating cameras
python3 script/calibrate.py

echo recitifying video
python3 script/rectify.py

echo tracking the fish
python3 script/get_shapes.py
python3 script/get_kernels.py
python3 script/get_features.py

echo linking the trajectories
python3 script/link.py

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
