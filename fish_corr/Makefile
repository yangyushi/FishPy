CPPFLAGS = -Ofast -Wall -I${prefix}/include/voro++ -std=c++11 -shared -I. -fPIC
LDLIBS = -L${prefix}/lib -lvoro++
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`

voro`${PY}-config --extension-suffix`: voro.cpp
	g++ $(LDLIBS) $(CPPFLAGS)  $(PYFLAGS) voro.cpp -o voro`${PY}-config --extension-suffix` 

tower_sample`python3-config --extension-suffix`: tower_sample.cpp
	g++ $(CPPFLAGS) $(PYFLAGS) $(EXTRAFLAGS) tower_sample.cpp -o tower_sample`python3-config --extension-suffix`
