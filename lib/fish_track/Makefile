EIGEN = ../../extern/eigen3
PYBIND11 = ../../extern/pybind11/include

CPPFLAGS = -Wall -I${EIGEN} -I${PYBIND11} -std=c++14 -fPIC -shared -O3\
		   -Wno-unknown-warning-option

PYFLAGS = $(shell ${PY}-config --include) $(shell ${PY}-config --ldflags)
PYEXT = $(shell ${PY}-config --extension-suffix)

all: nrook$(PYEXT)

nrook$(PYEXT): nrook.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) nrook.cpp -o nrook$(PYEXT)

.PHONY: all

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.a
