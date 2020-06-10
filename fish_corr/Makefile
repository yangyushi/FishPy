CPPFLAGS = -Ofast -Wall -I. -I${prefix}/include/voro++ -std=c++14 -fPIC -shared -mtune=native -march=native
LFLAG = -L${prefix}/lib -lvoro++
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`
PYEXT = $(shell ${PY}-config --extension-suffix)

all: voro$(PYEXT) tower_sample$(PYEXT)

voro$(PYEXT): voro.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) voro.cpp $(LFLAG) -o voro$(PYEXT)

tower_sample$(PYEXT): tower_sample.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) $(EXTRAFLAGS) tower_sample.cpp -o tower_sample$(PYEXT)

.PHONY: all
