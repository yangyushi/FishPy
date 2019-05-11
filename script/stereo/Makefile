# Makefile for Yushi Yang's homemade fish tracking prjoect
# Calibrate the cameras
# 	-> cameras.pkl
# Locating fish in 3D
#   -> positions.pkl
# Linking positions into trajectories
#   -> trajectories.pkl

VPATH=result

trajectories.pkl: positions.pkl
	python3 get_trajectories.py
positions.pkl: cameras.pkl
	python3 get_positions.py
cameras.pkl:
	python3 calibration.py
sort:
	./sort.sh
calib:
	rm -f cameras.pkl
	make cameras.pkl
pos:
	rm -f positions.pkl
	make positions.pkl
traj:
	rm -f trajectories.pkl
	make trajectories.pkl
clean:
	rm -f *.pkl
	rm -f *.npy
	rm -f *.pdf
	rm -f *.png
	rm -rf result
