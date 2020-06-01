#include "greta.h"
#include <iostream>

int main(){
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
    Trajs2DV3 trajs_2d_v3{trajs_v1, trajs_v2, trajs_v3};

    st::Link l0{0, 0, 0, 0};
    st::Link l1{1, 1, 1, 0};
    st::Link l2{2, 2, 2, 0};
    st::Link l3{0, 1, 2, 5};
    st::Links sl_f0{vector<st::Link>{l0, l1, l2}};
    st::Links sl_f1{vector<st::Link>{l0, l1, l2}};
    st::Links sl_f2{vector<st::Link>{l0, l2}};         // missing stereo link
    st::Links sl_f3{vector<st::Link>{l0, l1, l2, l3}}; // wrong stereo link here
    STLinks stereo_links{sl_f0, sl_f1, sl_f2, sl_f3};

    StereoTrajs stereo_trajs{trajs_2d_v3, stereo_links, frames_v3, 2.0};
    stereo_trajs.get_validate_trajs();

    for (auto view : stereo_trajs.labels_){
        for (auto frame : view){
            for (auto l : frame){
                cout << l << ", ";
            }
            cout << endl;
        }
        cout << endl;
    }

    cout << "Validate Trajectories:" << endl;
    int i = 0;
    for (auto t : stereo_trajs.trajs_){
        cout << "SL indices of #"  << ++i << " ("<< t.error() << ")" << endl;
        for (auto num : t.id_){
            cout << num << ", ";
        }
        cout << endl;
    }

    StereoTrajs stereo_trajs_opt = optimise_links_confined(stereo_trajs);

    cout << "After optimisation" << endl;
    i = 0;
    for (auto t : stereo_trajs_opt.trajs_){
        cout << "SL indices of #"  << ++i << " ("<< t.error() << ")" << endl;
        for (auto num : t.id_){
            cout << num << ", ";
        }
        cout << endl;
    }
}
