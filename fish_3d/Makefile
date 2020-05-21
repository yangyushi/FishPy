CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++11 -fPIC -shared -O3
LDLIBS = -L${prefix}/lib
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`
PYEXT = `${PY}-config --extension-suffix`

cray_trace$(PYEXT): cray_trace.cpp stereo.cpp
	g++ $(LDLIBS) $(CPPFLAGS) $(PYFLAGS) stereo.cpp cray_trace.cpp -o cray_trace$(PYEXT)

all: cray_trace.cpp
	g++ $(LDLIBS) $(CPPFLAGS) $(PYFLAGS) stereo.cpp cray_trace.cpp -o cray_trace$(PYEXT)

test:
	${PY} test.py
