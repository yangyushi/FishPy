VORO = ../../extern/voro++
CPPFLAGS = -Ofast -Wall -I. -I${VORO}/include/voro++ -std=c++14 -fPIC -shared -mtune=native -march=native
LFLAG = -L${VORO}/lib -lvoro++
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`
PYEXT = $(shell ${PY}-config --extension-suffix)

all: voro$(PYEXT) tower_sample$(PYEXT)

voro$(PYEXT): voro.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) voro.cpp $(LFLAG) -o voro$(PYEXT)

tower_sample$(PYEXT): tower_sample.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) tower_sample.cpp -o tower_sample$(PYEXT)

.PHONY: all


.clean:
	rm -f *.so
	rm -f *.o
	rm -f *.a
