#include "greta.h"


StereoTraj::StereoTraj(
        int k1, int k2, int k3,
        double c_max, int frames,
        const Trajs2DV3& trajs_2d_v3, const STLinks& links)
    : is_valid_{false}, c_max_{c_max}, st_indices_{}, st_errors_{},
      trajs_2d_{ trajs_2d_v3[0][k1], trajs_2d_v3[1][k2], trajs_2d_v3[2][k3]} {

        for (int frame = 0; frame < frames; frame++){
            bool link_found = false;
            array<int, 3> indices_v3{ trajs_2d_[0][frame], trajs_2d_[1][frame], trajs_2d_[2][frame] };

            int st_index = 0;
            for (auto link : links[frame].links_){
                if (indices_v3 == link.indices_){
                    is_valid_ = true;
                    link_found = true;
                    st_errors_.push_back(link.error_);
                    st_indices_.push_back(st_index);
                    break;
                }
                st_index++;
            }
            if (not link_found){
                st_errors_.push_back(c_max_);
                st_indices_.push_back(-1);
            }
        }
}


StereoTrajs::StereoTrajs(Trajs2DV3 trajs_2d_v3, STLinks links, double c_max, int frames)
    : c_max_{c_max}, frames_{frames} {
        for (int k1 = 0; k1 < trajs_2d_v3[0].size(); k1++){
        for (int k2 = 0; k2 < trajs_2d_v3[1].size(); k2++){
        for (int k3 = 0; k3 < trajs_2d_v3[2].size(); k3++){
                StereoTraj st{k1, k2, k3, c_max_, frames_, trajs_2d_v3, links};
                if (st.is_valid_){
                    trajs_.push_back(st);
                }
        }}}
}
