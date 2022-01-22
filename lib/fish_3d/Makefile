EIGEN = ../../extern/eigen3
PYBIND11 = ../../extern/pybind11/include

CPPFLAGS = -I${EIGEN} -I${PYBIND11} -std=c++14 -O3\
		   -I$(CPLEX_ROOT)/cplex/include\
		   -I$(CPLEX_ROOT)/concert/include\
		   -DIL_STD -I. -fPIC\
		   -Wno-unknown-warning-option

LFLAG=-L$(CPLEX_ROOT)/cplex/lib/$(CPLEX_ARCH)/static_pic\
	  -L/$(CPLEX_ROOT)/concert/lib/$(CPLEX_ARCH)/static_pic\
	  -lilocplex -lconcert -lcplex -lm -lpthread -ldl

PYFLAGS = $(shell ${PY}-config --includes) $(shell ${PY}-config --ldflags)
PYEXT = $(shell ${PY}-config --extension-suffix)

all: cgreta$(PYEXT) cstereo$(PYEXT) cray_trace$(PYEXT) ctemporal$(PYEXT) cutility$(PYEXT)

cgreta$(PYEXT): libgreta.a libtemporal.a libstereo.a cgreta.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) -shared cgreta.cpp libgreta.a libtemporal.a libstereo.a $(LFLAG) -o cgreta$(PYEXT)

ctemporal$(PYEXT): libtemporal.a ctemporal.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) -shared ctemporal.cpp libtemporal.a $(LFLAG) -o ctemporal$(PYEXT)

cstereo$(PYEXT): libstereo.a cstereo.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) -shared cstereo.cpp libstereo.a $(LFLAG) -o cstereo$(PYEXT)

cutility$(PYEXT): cutility.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) -shared cutility.cpp $(LFLAG) -o cutility$(PYEXT)

libgreta.a: greta.cpp greta.h
	$(CC) $(CPPFLAGS) -c greta.cpp -o greta.o
	ar rcs libgreta.a greta.o
	rm -f greta.o

libstereo.a: stereo.cpp stereo.h
	$(CC) $(CPPFLAGS) -c stereo.cpp -o stereo.o
	ar rcs libstereo.a stereo.o
	rm -f stereo.o

libtemporal.a: temporal.cpp temporal.h
	$(CC) $(CPPFLAGS) -c temporal.cpp -o temporal.o
	ar rcs libtemporal.a temporal.o
	rm -f temporal.o

cray_trace$(PYEXT): cray_trace.cpp
	$(CC) $(CPPFLAGS) $(PYFLAGS) -shared stereo.cpp cray_trace.cpp $(LFLAG) -o cray_trace$(PYEXT)

.PHONY: all

test: libgreta.a libtemporal.a libstereo.a
	$(CC) $(CPPFLAGS)  test_greta.cpp libtemporal.a libstereo.a libgreta.a $(LFLAG) -o test
	./test
	rm -f test

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.a
