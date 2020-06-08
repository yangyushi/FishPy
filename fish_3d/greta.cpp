#include "greta.h"


st::Coord3D StereoTraj::get_coordinates(
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
        ) const {
    int n_frames = frames_v3_[0].size();
    st::Coord3D result{n_frames, 3};
    array<st::Vec2D, 3> coordinates_2d;
    for (int t = 0; t < n_frames; t++){
        for (int view = 0; view < 3; view++){
            coordinates_2d[view] = frames_v3_[view][t].row(labels_[view][t]);
        }
        result.row(t) = st::three_view_reconstruct(coordinates_2d, Ps, Os);
    }

    return result;
}

StereoTraj::StereoTraj(
        int k1, int k2, int k3, double c_max,
        const FramesV3& frames_v3,
        const TemporalTrajs& temporal_trajs,
        const STLinks& links)
    : frames_v3_{frames_v3}, c_max_{c_max},
      is_valid_{false}, error_{0}, labels_{} {
        array<tp::Traj, 3> trajs_2d {
            temporal_trajs[0][k1], temporal_trajs[1][k2], temporal_trajs[2][k3]
        };
        array<int, 3> indices_v3;
        int st_index = 0;
        int frame_num = frames_v3_[0].size();

        array<st::Vec2D, 3> p_m1;  ///< p[-1]
        array<st::Vec2D, 3> p_0;  ///< p[-1]
        array<st::Vec2D, 3> v_mean;  ///< p[-2]

        for (int view = 0; view < 3; view++){
            pos_start_[view]   = frames_v3[view][      0      ].row(trajs_2d[view][      0      ]);
            p_m1[view]         = frames_v3[view][frame_num - 1].row(trajs_2d[view][frame_num - 1]);
            v_mean[view] = (p_m1[view] - pos_start_[view]) / (frame_num - 1);
            pos_predict_[view] = p_m1[view] + v_mean[view];
        }

        for (int frame = 0; frame < frame_num; frame++){
            bool link_found = false;
            for (int view = 0; view < 3; view++ ){
                indices_v3[view] = trajs_2d[view][frame];
                labels_[view].push_back(trajs_2d[view][frame]);
            }
            // find stereo link that link 2d trajectories in current frame
            st_index = 0;
            for (auto& link : links[frame].links_){
                if (indices_v3 == link.indices_){
                    is_valid_ = true;
                    link_found = true;
                    error_ += link.error_;
                    break;
                }
                st_index++;
            }
            if (not link_found){
                error_ += c_max_;
            }
        }
        error_ /= frame_num;
}

StereoTraj::StereoTraj(const StereoTraj& t, const FramesV3& frames_v3)
    : frames_v3_{frames_v3}, c_max_{t.c_max_}, is_valid_{t.is_valid_},
      error_{t.error_}, labels_{t.labels_}, pos_start_{t.pos_start_},
      pos_predict_{t.pos_predict_} {}


array<st::Coord2D, 3> StereoTraj::get_coordinates_2d() const {
    array<st::Coord2D, 3> result;
    unsigned long n_frames = frames_v3_[0].size();
    for (int view=0; view < 3; view++){
        st::Coord2D coord_2d{n_frames, 2};
        for (int t = 0; t < n_frames; t++){
            int id = labels_[view][t];
            coord_2d.row(t) = frames_v3_[view][t].row(id);
        }
        result[view] = coord_2d;
    }
    return result;
}

st::Coord2D StereoTraj::get_coordinates_2d(int view) const {
    unsigned long n_frames = frames_v3_[0].size();
    st::Coord2D result{n_frames, 2};
    for (int t = 0; t < n_frames; t++){
        int id = labels_[view][t];
        result.row(t) = frames_v3_[view][t].row(id);
    }
    return result;
}

StereoTrajs::StereoTrajs(
        TemporalTrajs temporal_trajs, STLinks links, FramesV3 frames_v3, double c_max
        )
    : temporal_trajs_{temporal_trajs}, st_links_{links},
      frames_v3_{frames_v3}, c_max_{c_max},
      labels_{}, size_{0} {
        int v = 0;
        for (auto& frames : frames_v3_){  ///< iter over views  -> vector< Coord2D >
            int f = 0;
            for (auto& frame : frames){  ///< iter over frames -> Coord2D
                labels_[v].push_back(set<int>{});
                for (int  i = 0; i < frame.rows(); i++){ ///< particle IDs
                    labels_[v][f].insert(i);
                }
                f++;
            }
            v++;
        }
}

StereoTrajs::StereoTrajs(const StereoTrajs& rhs)
    : temporal_trajs_{rhs.temporal_trajs_}, st_links_{rhs.st_links_},
      frames_v3_{rhs.frames_v3_}, c_max_{rhs.c_max_}, trajs_{}, size_{0} {
          for (auto& traj : rhs.trajs_){
              add(traj);
          }
}

vector<st::Coord3D> StereoTrajs::get_coordinates(
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
        ) const {
    vector<st::Coord3D> result;
    for (auto& traj : trajs_){
        result.push_back(traj.get_coordinates(Ps, Os));
    }
    return result;
}

void StereoTrajs::get_validate_trajs(){
        for (int k1 = 0; k1 < temporal_trajs_[0].size(); k1++){
        for (int k2 = 0; k2 < temporal_trajs_[1].size(); k2++){
        for (int k3 = 0; k3 < temporal_trajs_[2].size(); k3++){
                StereoTraj st{
                    k1, k2, k3, c_max_,
                    frames_v3_,
                    temporal_trajs_,
                    st_links_
                };
                if (st.is_valid_){
                    trajs_.push_back(st);
                    size_++;
                }
        }}}
}

void StereoTrajs::add(const StereoTraj& traj){
    if (traj.is_valid_){
        trajs_.push_back(StereoTraj{traj, this->frames_v3_});
        size_++;
    }
}

void StereoTrajs::clear(){
    trajs_.clear();
    size_ = 0;
}
