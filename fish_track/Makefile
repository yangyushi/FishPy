CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++14 -fPIC -shared -O3
LDLIBS = -L${prefix}/lib
PYFLAGS = $(shell ${PY} -m pybind11 --includes) $(shell ${PY}-config --ldflags)
PYEXT = $(shell ${PY}-config --extension-suffix)

all: nrook$(PYEXT)

nrook$(PYEXT): nrook.cpp
	$(CC) $(LDLIBS) $(CPPFLAGS) $(PYFLAGS) nrook.cpp -o nrook$(PYEXT)

.PHONY: all
