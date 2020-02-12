CPPFLAGS = -Ofast -Wall -I. -I${prefix}/include/voro++ -std=c++14 -fPIC -shared -mtune=native -march=native
LDLIBS = -L${prefix}/lib -lvoro++
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`
PYEXT = `${PY}-config --extension-suffix`

all: voro.cpp tower_sample.cpp
	g++ $(LDLIBS) $(CPPFLAGS)  $(PYFLAGS) voro.cpp -o voro`${PY}-config --extension-suffix` 
	g++ $(CPPFLAGS) $(PYFLAGS) $(EXTRAFLAGS) tower_sample.cpp -o tower_sample`${PY}-config --extension-suffix`

voro$(PYEXT): voro.cpp
	g++ $(LDLIBS) $(CPPFLAGS)  $(PYFLAGS) voro.cpp -o voro`${PY}-config --extension-suffix` 

tower_sample$(PYEXT): tower_sample.cpp
	g++ $(CPPFLAGS) $(PYFLAGS) $(EXTRAFLAGS) tower_sample.cpp -o tower_sample`${PY}-config --extension-suffix`
