#include "greta.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;


vector<st::Coord3D> get_trajs_3d(
        FramesV3 frames_v3, vector<st::PYLinks> stereo_links_py,
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os,
        double c_max, double search_range
        ){
    tp::Trajs trajs_v1 = tp::link_2d(frames_v3[0], search_range, false);
    tp::Trajs trajs_v2 = tp::link_2d(frames_v3[1], search_range, false);
    tp::Trajs trajs_v3 = tp::link_2d(frames_v3[2], search_range, false);

    cout << "tracking in 2D" << endl;
    Trajs2DV3 trajs_2d_v3{trajs_v1, trajs_v2, trajs_v3};

    cout << "initialising stereo links" << endl;
    STLinks stereo_links;
    for (auto links_py : stereo_links_py){
        stereo_links.push_back(st::Links{links_py});
    }
    cout << "initialising stereo trajectories" << endl;
    StereoTrajs stereo_trajs{trajs_2d_v3, stereo_links, frames_v3, c_max};
    stereo_trajs.get_validate_trajs();

    cout << "optimising stereo trajectories [" << stereo_trajs.trajs_.size() << "]" << endl;
    StereoTrajs stereo_trajs_opt = optimise_links_confined(stereo_trajs);

    cout << "optimisation finished [" << stereo_trajs_opt.trajs_.size() << "]" << endl;
    return stereo_trajs_opt.get_coordinates(Ps, Os);
}


PYBIND11_MODULE(cgreta, m){
    m.doc() = "GReTA tracking";
    m.def(
            "get_trajs_3d", &get_trajs_3d,
            py::arg("frames_v3"), py::arg("stereo_links"),
            py::arg("project_matrices"), py::arg("camera_origins"),
            py::arg("c_max"), py::arg("search_range")
            );
}
