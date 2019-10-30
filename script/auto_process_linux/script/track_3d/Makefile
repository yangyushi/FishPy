# Makefile for Yushi Yang's homemade fish tracking prjoect
# Calibrate the cameras
# 	-> cameras.pkl
# Locating fish in 3D
#   -> positions.pkl
# Linking positions into trajectories
#   -> trajectories.pkl

VPATH=result

positions.pkl:
	python3 get_positions.py
calib:
	python3 calibration.py
sort:
	./sort.sh
pos:
	rm -f positions.pkl >/dev/null
	rm -f result/positions.pkl >/dev/null
	make positions.pkl
epi:
	python3 check_epipolar.py
clean:
	rm -f *.pkl >/dev/null
	rm -f *.npy >/dev/null
	rm -f *.pdf >/dev/null
	rm -f *.png >/dev/null
	rm -rf result >/dev/null
