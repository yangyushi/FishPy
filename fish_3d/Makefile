CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++11 -fPIC -shared -Ofast -mtune=native -march=native
LDLIBS = -L${prefix}/lib
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`

cray_trace`${PY}-config --extension-suffix`: cray_trace.cpp
	g++ -o cray_trace`${PY}-config --extension-suffix` cray_trace.cpp $(LDLIBS) $(CPPFLAGS) $(PYFLAGS)

test:
	${PY} test.py
