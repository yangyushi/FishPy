CPPFLAGS = -Wall -I/usr/local/include/eigen3 -std=c++11 -fPIC -shared
LDLIBS = -L/usr/local/lib
PYFLAGS = `python3 -m pybind11 --includes` `python3-config --ldflags`

cray_trace`python3-config --extension-suffix`: cray_trace.cpp
	g++ -o cray_trace`python3-config --extension-suffix` cray_trace.cpp $(LDLIBS) $(CPPFLAGS) $(PYFLAGS)
