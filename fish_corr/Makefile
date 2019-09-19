CPPFLAGS = -Wall -I/usr/local/include/voro++ -std=c++11 -shared -I. -fPIC
LDLIBS = -L/usr/local/lib -lvoro++
PYFLAGS = `python3 -m pybind11 --includes` `python3-config --ldflags`

voro`python3-config --extension-suffix`: voro.cpp
	g++ -o voro`python3-config --extension-suffix` voro.cpp $(LDLIBS) $(CPPFLAGS)  $(PYFLAGS)$
