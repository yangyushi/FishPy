#include "greta.h"
#include <iostream>

int main(){

    const bool VERBOSE = false;

    tp::Coord2D f0{3, 2};
    tp::Coord2D f1{3, 2};
    tp::Coord2D f2{3, 2};
    tp::Coord2D f3{3, 2};
    f0 << 0.0, 0.0, 0.0, 2.0, 0.0, 4.0;
    f1 << 0.0, 0.2, 0.0, 2.2, 0.0, 4.2;
    f2 << 0.0, 0.4, 0.0, 2.4, 0.0, 4.4;
    f3 << 0.0, 0.5, 0.0, 2.5, 0.0, 4.5;
    vector<temporal::Coord2D> frames {f0, f1, f2, f3};
    FramesV3 frames_v3{frames, frames, frames};

    tp::Trajs trajs_v1 = tp::link_2d(frames, 2.4, false);
    tp::Trajs trajs_v2 = tp::link_2d(frames, 2.4, false);
    tp::Trajs trajs_v3 = tp::link_2d(frames, 2.4, false);
    TemporalTrajs temporal_trajs{trajs_v1, trajs_v2, trajs_v3};

    st::Link l0{0, 0, 0, 0};
    st::Link l1{1, 1, 1, 0};
    st::Link l2{2, 2, 2, 0};
    st::Link l3{0, 1, 2, 5};
    st::Links sl_f0{vector<st::Link>{l0, l1, l2}};
    st::Links sl_f1{vector<st::Link>{l0, l1, l2}};
    st::Links sl_f2{vector<st::Link>{l0, l2}};         // missing stereo link
    st::Links sl_f3{vector<st::Link>{l0, l1, l2, l3}}; // wrong stereo link here
    STLinks stereo_links{sl_f0, sl_f1, sl_f2, sl_f3};

    StereoTrajs stereo_trajs{temporal_trajs, stereo_links, frames_v3, 2.0};
    stereo_trajs.get_validate_trajs();

    if (VERBOSE){
        cout << "Validate Trajectories:" << endl;
        int i = 0;
        for (auto t : stereo_trajs.trajs_){
            cout << "SL indices of #"  << ++i << " ("<< t.error_ << ")" << endl;
            for (auto num : t.id_){
                cout << num << ", ";
            }
            cout << endl;
        }
    }

    StereoTrajs stereo_trajs_opt = optimise_links_confined(stereo_trajs);

    if (VERBOSE){
        cout << "After optimisation" << endl;
        int i = 0;
        for (auto t : stereo_trajs_opt.trajs_){
            cout << "SL indices of #"  << ++i << " ("<< t.error_ << ")" << endl;
            for (auto num : t.id_){
                cout << num << ", ";
            }
            cout << endl;
        }
    }

    cout << "Creating Meta Objects (LV1), near root? ";

    vector<StereoTrajs> meta_lv1 {stereo_trajs_opt, stereo_trajs_opt, stereo_trajs_opt};

    STLinks stereo_links_meta = get_meta_stereo_links(meta_lv1);
    MetaFramesV3 frames_meta = get_meta_frames(meta_lv1);

    TemporalTrajs temporal_trajs_meta{
        tp::link_meta(frames_meta[0], 20, false),
        tp::link_meta(frames_meta[1], 20, false),
        tp::link_meta(frames_meta[2], 20, false),
    };

    MetaSTs<StereoTrajs> meta_sts_lv1 {
        temporal_trajs_meta, stereo_links_meta, frames_meta, meta_lv1, 2.0
    };

    cout << meta_sts_lv1.near_root_ << endl;

    meta_sts_lv1.get_validate_trajs();
    MetaSTs<StereoTrajs> meta_sts_lv1_opt = optimise_links_confined(meta_sts_lv1);

    if (VERBOSE){
        cout << "After optimisation Meta Stereo Trajectories" << endl;
        int i = 0;
        for (auto t : meta_sts_lv1_opt.trajs_){
            cout << "ID of #"  << ++i << " ("<< t.error_ << ")" << endl;
            for (auto num : t.id_){
                cout << num << ", ";
            }
            cout << endl;
        }
    }

    cout << "Total Frames: " << meta_sts_lv1_opt.get_total_frames() << endl;

    cout << "Creating Meta Objects (LV2), near root? ";

    vector< MetaSTs<StereoTrajs> > meta_lv2 {meta_sts_lv1_opt, meta_sts_lv1_opt, meta_sts_lv1_opt};


    STLinks stereo_links_meta_lv2 = get_meta_stereo_links(meta_lv2);
    MetaFramesV3 frames_meta_lv2 = get_meta_frames(meta_lv2);

    TemporalTrajs temporal_trajs_meta_lv2{
        tp::link_meta(frames_meta_lv2[0], 20, false),
        tp::link_meta(frames_meta_lv2[1], 20, false),
        tp::link_meta(frames_meta_lv2[2], 20, false),
    };

    MetaSTs< MetaSTs<StereoTrajs> > meta_sts_lv2 {
        temporal_trajs_meta_lv2, stereo_links_meta_lv2, frames_meta_lv2, meta_lv2, 2.0
    };

    cout << meta_sts_lv2.near_root_ << endl;

    meta_sts_lv2.get_validate_trajs();
    MetaSTs< MetaSTs<StereoTrajs> > meta_sts_lv2_opt = optimise_links_confined(meta_sts_lv2);

    cout << "Total Frames: " << meta_sts_lv2_opt.get_total_frames() << endl;

    cout << "Creating Meta Objects (LV3), near root? ";

    vector< MetaSTs< MetaSTs<StereoTrajs> > > meta_lv3{
        meta_sts_lv2_opt, meta_sts_lv2_opt, meta_sts_lv2_opt
    };
    STLinks stereo_links_meta_lv3 = get_meta_stereo_links(meta_lv3);
    MetaFramesV3 frames_meta_lv3 = get_meta_frames(meta_lv3);
    TemporalTrajs temporal_trajs_meta_lv3{
        tp::link_meta(frames_meta_lv3[0], 20, false),
        tp::link_meta(frames_meta_lv3[1], 20, false),
        tp::link_meta(frames_meta_lv3[2], 20, false),
    };
    MetaSTs < MetaSTs< MetaSTs< StereoTrajs> > > meta_sts_lv3 {
        temporal_trajs_meta_lv3, stereo_links_meta_lv3, frames_meta_lv3, meta_lv3, 2.0
    };

    cout << meta_sts_lv3.near_root_ << endl;

    meta_sts_lv3.get_validate_trajs();
    MetaSTs< MetaSTs< MetaSTs< StereoTrajs> > > meta_sts_lv3_opt = optimise_links_confined(meta_sts_lv3);

    cout << "Total Frames: " << meta_sts_lv3_opt.get_total_frames() << endl;

    st::ProjMat P;
    P << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0;
    st::Vec3D O;
    O << 0, 0, 0;
    array<st::ProjMat, 3> Ps {P, P, P};
    array<st::Vec3D, 3> Os {O, O, O};
    vector<st::Coord3D> trajs = meta_sts_lv3_opt.get_coordinates(Ps, Os);
    for (auto t : trajs){
        cout << t.transpose().leftCols(8) << endl;
    }
}
