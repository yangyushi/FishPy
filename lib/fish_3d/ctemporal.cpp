#include "temporal.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace temporal;

PYTrajs get_trajectories(vector<Coord2D> frames, double search_range, bool allow_fragment){
    PYTrajs result;
    Trajs trajs = link_2d(frames, search_range, allow_fragment);
    for (auto t : trajs){
        result.push_back( PYTraj{t.indices_, t.time_} );
    }
    return result;
}

PYBIND11_MODULE(ctemporal, m){
    m.doc() = "a 2D linking modules";
    m.def( "get_trajectories", &get_trajectories,
            py::arg("frames"), py::arg("search_range"), py::arg("allow_fragment") );
}
