#include "stereo.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace stereo;


PYLinks match_v3(
        Coord2D& centres_1, Coord2D& centres_2, Coord2D& centres_3,
        ProjMat P1, ProjMat P2, ProjMat P3,
        Vec3D O1, Vec3D O2, Vec3D O3, double tol_2d, bool optimise
        ){

    Links links = three_view_match(
            centres_1,  centres_2,  centres_3,
            P1,  P2,  P3,
            O1,  O2,  O3,
            tol_2d, optimise
            );

    return links.to_py();
}


Coord2D refractive_project(Coord3D& points, ProjMat& P, Vec3D& O){
    Vec3D xyz;
    Coord2D result{points.rows(), 2};
    for (int i = 0; i < points.rows(); i++){
        xyz = points.row(i);
        result.row(i) = reproject_refractive(xyz, P, O);
    }
    return result;
}



PYBIND11_MODULE(cstereo, m){
    m.doc() = "stereo matching & greta tracking";
    m.def(
            "match_v3", &match_v3,
            py::arg("centres_1"), py::arg("centres_2"), py::arg("centres_3"),
            py::arg("P1"), py::arg("P2"), py::arg("P3"),
            py::arg("O1"), py::arg("O2"), py::arg("O3"),
            py::arg("tol_2d"), py::arg("optimise")=true
            );
    m.def(
            "get_error", &get_error,
            py::arg("centres"), py::arg("Ps"), py::arg("Os")
            );
    m.def(
            "refractive_project", &refractive_project,
            py::arg("points"), py::arg("P"), py::arg("O")
            );
}
