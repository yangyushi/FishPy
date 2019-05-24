g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I/usr/local/include ray_trace.cpp -o ray_trace`python3-config --extension-suffix` `python3-config --ldflags`
