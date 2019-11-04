CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++14 -fPIC -shared -Ofast -mtune=native -march=native
LDLIBS = -L${prefix}/lib
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`

nrook`${PY}-config --extension-suffix`: nrook.cpp
	g++ -o nrook`${PY}-config --extension-suffix` nrook.cpp $(LDLIBS) $(CPPFLAGS) $(PYFLAGS)

test:
	${PY} test.py
