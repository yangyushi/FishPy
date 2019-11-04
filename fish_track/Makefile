CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++14 -fPIC -shared -Ofast -mtune=native -march=native
LDLIBS = -L${prefix}/lib
PYFLAGS = `${PY} -m pybind11 --includes` `${PY}-config --ldflags`

nrook`${PY}-config --extension-suffix`: nrook.cpp
	g++ $(LDLIBS) $(CPPFLAGS) $(PYFLAGS) nrook.cpp -o nrook`${PY}-config --extension-suffix` 

test:
	${PY} test.py
