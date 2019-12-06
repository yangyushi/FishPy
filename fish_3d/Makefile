CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++11 -fPIC -shared -Ofast -mtune=native -march=native
LDLIBS = -L${prefix}/lib
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`
PYEXT = `${PY}-config --extension-suffix`


cray_trace$(PYEXT): cray_trace.cpp
	g++ $(LDLIBS) $(CPPFLAGS) $(PYFLAGS) cray_trace.cpp -o cray_trace`${PY}-config --extension-suffix` 

all: cray_trace.cpp
	g++ $(LDLIBS) $(CPPFLAGS) $(PYFLAGS) cray_trace.cpp -o cray_trace`${PY}-config --extension-suffix` 

test:
	${PY} test.py
