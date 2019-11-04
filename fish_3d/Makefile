CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++11 -fPIC -shared -Ofast -mtune=native -march=native
LDLIBS = -L${prefix}/lib
PYFLAGS = `python3 -m pybind11 --includes` `python3-config --ldflags`

cray_trace`python3-config --extension-suffix`: cray_trace.cpp
	g++ -o cray_trace`python3-config --extension-suffix` cray_trace.cpp $(LDLIBS) $(CPPFLAGS) $(PYFLAGS)

test:
	python3 test.py
