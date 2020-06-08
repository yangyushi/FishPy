#include "greta.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;


vector< tuple<st::Coord3D, double> > get_trajs_3d(
        FramesV3 frames_v3, vector<st::PYLinks> stereo_links_py,
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os,
        double c_max, double search_range
        ){
    vector< tuple<st::Coord3D, double> > result;

    tp::Trajs trajs_v1 = tp::link_2d(frames_v3[0], search_range, false);
    tp::Trajs trajs_v2 = tp::link_2d(frames_v3[1], search_range, false);
    tp::Trajs trajs_v3 = tp::link_2d(frames_v3[2], search_range, false);

    TemporalTrajs trajs_2d_v3{trajs_v1, trajs_v2, trajs_v3};

    STLinks stereo_links;
    for (auto links_py : stereo_links_py){
        stereo_links.push_back(st::Links{links_py});
    }
    StereoTrajs stereo_trajs{trajs_2d_v3, stereo_links, frames_v3, c_max};
    stereo_trajs.get_validate_trajs();

    StereoTrajs stereo_trajs_opt = optimise_links_confined(stereo_trajs);

    auto trajs_3d = stereo_trajs_opt.get_coordinates(Ps, Os);
    for (int i = 0; i < trajs_3d.size(); i++){
        auto traj = trajs_3d[i];
        double error = stereo_trajs_opt.trajs_[i].error_;
        result.push_back(make_tuple(traj, error));
    }
    return result;
}


vector< tuple<st::Coord3D, double> > get_trajs_3d_t1t2(
        FramesV3 frames_v3, vector<st::PYLinks> stereo_links_py,
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os,
        double c_max, double search_range, double search_range_traj,
        int tau_1, int tau_2
        ){
    if (tau_1 * tau_2 != frames_v3[0].size()){
        throw runtime_error("Invalid time division (τ1 * τ2 != T)");
    }

    vector< tuple<st::Coord3D, double> > result;

    vector<StereoTrajs> meta_lv1;
    for (int t2 = 0; t2 < tau_2; t2++){
        FramesV3 sub_frames_v3;
        for (int view = 0; view < 3; view++){
            for (int t = t2 * tau_1; t < t2 * tau_1 + tau_1; t++){
                sub_frames_v3[view].push_back(frames_v3[view][t]);
            }
        }

        TemporalTrajs temporal_trajs;

        for (int view = 0; view < 3; view++){
            temporal_trajs[view] = tp::link_2d(sub_frames_v3[view], search_range, false);
        }

        STLinks stereo_links;
        for (int t = t2 * tau_1; t < t2 * tau_1 + tau_1; t++){
            stereo_links.push_back(st::Links{stereo_links_py[t]});
        }

        StereoTrajs stereo_trajs{temporal_trajs, stereo_links, sub_frames_v3, c_max};
        stereo_trajs.get_validate_trajs();

        StereoTrajs stereo_trajs_opt = optimise_links_confined(stereo_trajs);

        meta_lv1.push_back(stereo_trajs_opt);
    }

    STLinks stereo_links_meta = get_meta_stereo_links(meta_lv1);
    MetaFramesV3 frames_meta = get_meta_frames(meta_lv1);

    TemporalTrajs temporal_trajs_meta;
    for (int view = 0; view < 3; view++){
        temporal_trajs_meta[view] = tp::link_meta(frames_meta[view], search_range_traj, false);
    }

    MetaSTs<StereoTrajs> meta_sts_lv1 {
        temporal_trajs_meta, stereo_links_meta, frames_meta, meta_lv1, c_max
    };

    meta_sts_lv1.get_validate_trajs();


    MetaSTs<StereoTrajs> meta_sts_lv1_opt = optimise_links_confined(meta_sts_lv1);

    auto trajs_3d = meta_sts_lv1_opt.get_coordinates(Ps, Os);

    for (int i = 0; i < trajs_3d.size(); i++){
        auto traj = trajs_3d[i];
        double error = meta_sts_lv1_opt.trajs_[i].error_;
        result.push_back(make_tuple(traj, error));
    }

    return result;
}


vector< tuple<st::Coord3D, double> > get_trajs_3d_t1t2t3(
        FramesV3 frames_v3, vector<st::PYLinks> stereo_links_py,
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os,
        double c_max, double search_range, double search_range_traj,
        int tau_1, int tau_2, int tau_3
        ){
    if (tau_1 * tau_2 * tau_3 != frames_v3[0].size()){
        throw runtime_error("Invalid time division (τ1 * τ2 * τ3 != T)");
    }

    vector< tuple<st::Coord3D, double> > result;

    vector< MetaSTs<StereoTrajs> > meta_lv2;
    for (int t3 = 0; t3 < tau_3; t3++){
        vector<StereoTrajs> meta_lv1;
        for (int t2 = 0; t2 < tau_2; t2++){
            int t0  = t3 * (tau_1 * tau_2) + t2 * tau_1;
            FramesV3 sub_frames_v3;
            for (int view = 0; view < 3; view++){
                for (int t = t0; t < t0 + tau_1; t++){
                    sub_frames_v3[view].push_back(frames_v3[view][t]);
                }
            }
            TemporalTrajs temporal_trajs;
            for (int view = 0; view < 3; view++){
                temporal_trajs[view] = tp::link_2d(sub_frames_v3[view], search_range, false);
            }

            STLinks stereo_links;
            for (int t = t0; t < t0 + tau_1; t++){
                stereo_links.push_back(st::Links{stereo_links_py[t]});
            }

            StereoTrajs stereo_trajs{temporal_trajs, stereo_links, sub_frames_v3, c_max};
            stereo_trajs.get_validate_trajs();

            StereoTrajs stereo_trajs_opt = optimise_links_confined(stereo_trajs);

            meta_lv1.push_back(stereo_trajs_opt);
        }

        STLinks stereo_links_meta_lv1 = get_meta_stereo_links(meta_lv1);
        MetaFramesV3 frames_meta_lv1 = get_meta_frames(meta_lv1);

        TemporalTrajs temporal_trajs_meta_lv1;
        for (int view = 0; view < 3; view++){
            temporal_trajs_meta_lv1[view] = tp::link_meta(frames_meta_lv1[view], search_range_traj, false);
        }

        MetaSTs<StereoTrajs> meta_sts_lv1 {
            temporal_trajs_meta_lv1, stereo_links_meta_lv1, frames_meta_lv1, meta_lv1, c_max
        };

        meta_sts_lv1.get_validate_trajs();

        MetaSTs<StereoTrajs> meta_sts_lv1_opt = optimise_links_confined(meta_sts_lv1);

        meta_lv2.push_back(meta_sts_lv1_opt);
    }

    STLinks stereo_links_meta_lv2 = get_meta_stereo_links(meta_lv2);
    MetaFramesV3 frames_meta_lv2 = get_meta_frames(meta_lv2);

    TemporalTrajs temporal_trajs_meta_lv2;
    for (int view = 0; view < 3; view++){
        temporal_trajs_meta_lv2[view] = tp::link_meta(frames_meta_lv2[view], search_range_traj, false);
    }

    MetaSTs<MetaSTs <StereoTrajs> > meta_sts_lv2 {
        temporal_trajs_meta_lv2, stereo_links_meta_lv2, frames_meta_lv2, meta_lv2, c_max
    };

    meta_sts_lv2.get_validate_trajs();

    MetaSTs< MetaSTs<StereoTrajs> > meta_sts_lv2_opt = optimise_links_confined(meta_sts_lv2);

    auto trajs_3d = meta_sts_lv2_opt.get_coordinates(Ps, Os);

    for (int i = 0; i < trajs_3d.size(); i++){
        auto traj = trajs_3d[i];
        double error = meta_sts_lv2_opt.trajs_[i].error_;
        result.push_back(make_tuple(traj, error));
    }

    return result;
}


PYBIND11_MODULE(cgreta, m){
    m.doc() = "GReTA tracking";
    m.def(
            "get_trajs_3d", &get_trajs_3d,
            py::arg("frames_v3"), py::arg("stereo_links"),
            py::arg("project_matrices"), py::arg("camera_origins"),
            py::arg("c_max"), py::arg("search_range")
        );
    m.def(
            "get_trajs_3d_t1t2", &get_trajs_3d_t1t2,
            py::arg("frames_v3"), py::arg("stereo_links"),
            py::arg("project_matrices"), py::arg("camera_origins"),
            py::arg("c_max"), py::arg("search_range"), py::arg("search_range_traj"),
            py::arg("tau_1"), py::arg("tau_2")
        );
    m.def(
            "get_trajs_3d_t1t2t3", &get_trajs_3d_t1t2t3,
            py::arg("frames_v3"), py::arg("stereo_links"),
            py::arg("project_matrices"), py::arg("camera_origins"),
            py::arg("c_max"), py::arg("search_range"), py::arg("search_range_traj"),
            py::arg("tau_1"), py::arg("tau_2"), py::arg("tau_3")
        );
}
