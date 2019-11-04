CPPFLAGS = -Wall -I${prefix}/include/eigen3 -std=c++14 -fPIC -shared -Ofast -mtune=native -march=native
LDLIBS = -L${prefix}/lib
PYFLAGS = `python3 -m pybind11 --includes` `python3-config --ldflags`

nrook`python3-config --extension-suffix`: nrook.cpp
	g++ -o nrook`python3-config --extension-suffix` nrook.cpp $(LDLIBS) $(CPPFLAGS) $(PYFLAGS)

test:
	python3 test.py
