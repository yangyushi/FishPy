CPPFLAGS = -Ofast -Wall -I${prefix}/include/voro++ -std=c++11 -shared -I. -fPIC
LDLIBS = -L${prefix}/lib -lvoro++
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`

voro`${PY}-config --extension-suffix`: voro.cpp
	g++ $(LDLIBS) $(CPPFLAGS)  $(PYFLAGS)$ voro.cpp -o voro`${PY}-config --extension-suffix` 
