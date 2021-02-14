# Makefile for Yushi Yang's homemade fish tracking prjoect
# calculate the averaged image as background
# 	-> background.npy
# use segmentation to find fish shape
#	-> fish_shape_collection.npy
# use PCA to find kernels of shapes
# 	-> shape_kernels.npy
# use template matching to find feature maxima
#   -> maxima_*.npy

VPATH=result

features.pkl: shape_kernels.npy
	python3 get_features.py
shape_kernels.npy: fish_shape_collection.npy
	python3 get_kernels.py
fish_shape_collection.npy:
	python3 get_shapes.py
sort:
	./sort.sh
feature:
	rm -f features.pkl 2>/dev/null
	rm -f result/features.pkl 2>/dev/null
	make features.pkl
shape:
	rm -f fish_shape_collection.npy 2>/dev/null
	rm -f result/fish_shape_collection.npy 2>/dev/null
	make fish_shape_collection.npy 
kernel:
	rm -f shape_kernels.npy 2>/dev/null
	rm -f result/shape_kernels.npy 2>/dev/null
	make shape_kernels.npy 
roi:
	python3 get_roi.py
threshold:
	python3 get_threshold.py
clean:
	rm -f *.pkl 2>/dev/null
	rm -f *.npy 2>/dev/null
	rm -f *.pdf 2>/dev/null
	rm -f *.png 2>/dev/null
	rm -rf result 2>/dev/null
