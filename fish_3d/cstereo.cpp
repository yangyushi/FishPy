#include "stereo.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace stereo;
using Arr1D = Eigen::Array<double, Eigen::Dynamic, 1>;


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


tuple<PYLinks, Coord3D, Arr1D> match_v3_verbose(
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
    Coord3D positions_3d{links.links_.size(), 3};
    Vec3D xyz;
    Arr1D errors{links.links_.size(), 1};
    TriPM Ps{P1, P2, P3};
    TriXYZ Os{O1, O2, O3};
    TriXY centres_matched ;
    int index = 0;
    for (auto link : links.links_){
        centres_matched[0] = centres_1.row(link[0]);
        centres_matched[1] = centres_2.row(link[1]);
        centres_matched[2] = centres_3.row(link[2]);
        xyz = three_view_reconstruct(centres_matched, Ps, Os);
        errors.row(index) = get_error_with_xyz(centres_matched, Ps, Os, xyz);
        positions_3d.row(index) = xyz;
        index++;
    };
    return tuple<PYLinks, Coord3D, Arr1D>{
        links.to_py(), positions_3d, errors
    };
}


tuple<Coord3D, Arr1D> locate_v3(
            Coord2D& centres_1, Coord2D& centres_2, Coord2D& centres_3,
            ProjMat P1, ProjMat P2, ProjMat P3,
            Vec3D O1, Vec3D O2, Vec3D O3, double tol_2d, bool optimise
        ){
    auto result = three_view_match_verbose(
            centres_1,  centres_2,  centres_3,
            P1,  P2,  P3,
            O1,  O2,  O3,
            tol_2d, optimise
        );
    auto positions = get<1>(result); 
    auto errors = get<2>(result);

    Coord3D positions_arr{positions.size(), 3};

    Arr1D errors_arr{positions.size(), 1};
    for (int i = 0; i < positions.size(); i++){
        positions_arr.row(i) = positions[i];
        errors_arr.row(i) = errors[i];
    }

    return make_tuple(positions_arr, errors_arr);
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
            "match_v3_verbose", &match_v3_verbose,
            py::arg("centres_1"), py::arg("centres_2"), py::arg("centres_3"),
            py::arg("P1"), py::arg("P2"), py::arg("P3"),
            py::arg("O1"), py::arg("O2"), py::arg("O3"),
            py::arg("tol_2d"), py::arg("optimise")=true
            );
    m.def(
            "locate_v3", &locate_v3,
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
