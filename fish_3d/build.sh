g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I/usr/local/include cray_trace.cpp -o cray_trace`python3-config --extension-suffix` `python3-config --ldflags`
