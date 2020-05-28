#include "greta.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;


PYBIND11_MODULE(cgreta, m){
    m.doc() = "GReTA tracking";
}
