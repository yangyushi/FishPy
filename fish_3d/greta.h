#ifndef GRETA
#define GRETA
#include <ilcplex/ilocplex.h>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <set>
#include <map>
#include "temporal.h"
#include "stereo.h"

ILOSTLBEGIN  ///< marco from CPLEX for cross-platform compability
using namespace std;

namespace tp = temporal;
namespace st = stereo;


/*
 * 2D Temporal Trajectories in different views
 *    "shape" (3, n_trajs, n_frame, 2)
 *    the last axis is (indices, time)
 */
using TemporalTrajs =  array<tp::Trajs, 3>;

/*
 * A collections of meta particles
 *     - shape (3, n_frames, n_particles, 2)
 *     - a meta particle contains
 *          1. its index in the frame
 *          2. its initial 2d position
 *          3. the predicted 2d position in its last frame + 1
 *     - the order of the meta particles in a frame is determined
 *       by the order of its parent (meta)stereo-trajectory
 */
using MetaFramesV3 = array<tp::MetaFrames, 3>;

using STIndices = vector<int>;
using Labels = vector< set<int> >;    ///< shape (n_frames, n_particles)
using LabelsV3 = array<Labels  , 3>;  ///< shape (3, n_frames, n_particles)
using Frames = vector< tp::Coord2D >; ///< shape (n_frames, n_particles, 2)
using FramesV3 = array<Frames, 3>;    ///< shape (3, n_frames, n_particles, 2)
using STLinks = vector<st::Links>;    ///< shape (n_frame, n_links, 3)


/**
 * γ(t)
 * k1, k2, k3 are indices of 2d trajectories
 */
struct StereoTraj{
    array<int, 3> id_;
    const FramesV3 frames_v3_;

    bool is_valid_;  ///< false if no stereo link in any frame
    double c_max_;
    double error_; ///< The time-averaged reprojection error

    array<vector<int>, 3> labels_;  ///< labels[view][frame] --> particle id

    array<tp::Vec2D, 3> pos_start_;
    array<tp::Vec2D, 3> pos_predict_;

    inline bool contains_particle(int view, int frame, int id){
        return labels_[view][frame] == id;
    }
    /**
     * Calculate the 3D coordinates of this trajectory
     * @param Ps: projection matrices of three cameras
     * @param Os: origins of three cameras
     */
    st::Coord3D get_coordinates(array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os);
    StereoTraj(
            int k1, int k2, int k3, double c_max,
            const FramesV3 frames,
            const TemporalTrajs& trajs_2d_v3,
            const STLinks& links
            );
};

/**
 * { γ(t) }
 * All the 2d trajectories should have the same length in time
 */
struct StereoTrajs{
    TemporalTrajs temporal_trajs_; ///< (3, n_trajs, n_frame, 2)
    STLinks st_links_;             ///< (n_frame, n_links, 3)
    FramesV3 frames_v3_;           ///< (3, n_frames, n_particles, 2)
    LabelsV3 labels_;              ///< (3, n_frames, n_particles)
    double c_max_;
    int size_;                     ///< number of stereo trajs, trajs_.size()
    vector<StereoTraj> trajs_;     ///< (n_trajs, )
    const bool is_root_ = true;
    const bool near_root_ = false;

    vector<st::Coord3D> get_coordinates(
            array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
            );

    inline void get_total_frames(unsigned long& frame_num){
        frame_num *= frames_v3_[0].size();
    }

    inline unsigned long get_total_frames(){
        return frames_v3_[0].size();
    }

    void get_validate_trajs();
    void add(StereoTraj traj);
    void clear();

    StereoTrajs(
            TemporalTrajs temporal_trajs, STLinks links, FramesV3 frames,
            double c_max
            );
};

/**
 * the meta version of StereoTraj
 */
template<class T>
struct MetaST{
    array<int, 3> id_;
    vector<T>& parents_;
    const MetaFramesV3& frames_v3_;
    unsigned long total_frame_;
    double c_max_;
    double error_{0};
    bool is_valid_{false};
    array<vector<int>, 3> labels_{}; ///< (3, n_frame)
    array<tp::Vec2D, 3> pos_start_;
    array<tp::Vec2D, 3> pos_predict_;
    inline bool contains_particle(int view, int frame, int id){
        return labels_[view][frame] == id;
    }

    st::Coord3D get_coordinates(array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os);

    MetaST(
            int k1, int k2, int k3, double c_max, unsigned long total_frame,
            const MetaFramesV3& frames_v3,
            vector<T>& parents,
            const TemporalTrajs& trajs_2d_v3,
            const STLinks& links
            );
};

template<class T>
MetaST<T>::MetaST(
        int k1, int k2, int k3, double c_max, unsigned long total_frame,
        const MetaFramesV3& frames_v3,
        vector<T>& parents,
        const TemporalTrajs& temporal_trajs,
        const STLinks& links
        )
    : id_{k1, k2, k3}, parents_{parents}, frames_v3_{frames_v3},
      total_frame_{total_frame}, c_max_{c_max} {
        array<tp::Traj, 3> trajs_2d { ///< trajectories of meta particles belonging to
            temporal_trajs[0][k1],    //     this MetaST instance
            temporal_trajs[1][k2],    //     shape (3, n_frame,)
            temporal_trajs[2][k3]
        };
        array<int, 3> indices_v3;
        int st_index = 0, frame_num = frames_v3[0].size();

        array<int, 3> ids {k1, k2, k3};
        for (int view = 0; view < 3; view++){
            pos_start_[view]   = frames_v3[view][0]            [ids[view]][0];
            pos_predict_[view] = frames_v3[view][frame_num - 1][ids[view]][1];
        }

        for (int frame = 0; frame < frame_num; frame++){
            bool link_found = false;
            for (int view = 0; view < 3; view++ ){
                indices_v3[view] = trajs_2d[view][frame];
                labels_[view].push_back(trajs_2d[view][frame]);
            }
            st_index = 0;
            // find stereo link that link 2d trajectories in current frame
            for (auto link : links[frame].links_){
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

template<class T>
st::Coord3D MetaST<T>::get_coordinates(
            array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
        ){
    st::Coord3D result{total_frame_, 3};

    unsigned long n_frames = frames_v3_[0].size();
    const unsigned long block_size = parents_[0].get_total_frames();
    array<int, 3> meta_id;
    bool is_valid = false;

    for (int t = 0; t < n_frames; t++){
        is_valid = false;
        int t_shift = t * block_size;
        for (int view = 0; view < 3; view++){
            meta_id[view] = labels_[view][t];  ///< particle id in the meta frame
        }
        for (int idx = 0; idx < parents_[t].trajs_.size(); idx++){

            if (meta_id == array<int, 3>{idx, idx, idx}){

                st::Coord3D tmp = parents_[t].trajs_[idx].get_coordinates(Ps, Os);

                result.block(
                        t_shift, 0, block_size, 3
                    ) = parents_[t].trajs_[idx].get_coordinates(Ps, Os);

                is_valid = true;
                break;
            }
        }
        if (not is_valid){
            result.block(t_shift, 0, block_size, 3) = st::Coord3D::Constant(block_size, 3, NAN);
        }
    }
    return result;
}

template<class T>
struct MetaSTs{
    TemporalTrajs temporal_trajs_;  ///< (3, n_trajs, n_frame, 2)
    STLinks st_links_;             ///< (n_frame, n_links, 3)
    MetaFramesV3 frames_v3_;        ///< (3, n_frames, n_particles, 2, 2)
    vector<T> parents_;              ///< (n_frames, )
    LabelsV3 labels_;               ///< (3, n_frames, n_particles)
    double c_max_;
    int size_;                      ///< number of stereo trajs, trajs_.size()
    bool near_root_;                ///< if the parent is StereoTrajs
    vector< MetaST<T> > trajs_;
    const bool is_root_ = false;
    void get_validate_trajs();
    void add(MetaST<T> traj);
    void clear();
    unsigned long get_total_frames();
    void get_total_frames(unsigned long& frame_num);
    vector<st::Coord3D> get_coordinates(
            array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
            );
    MetaSTs(
            TemporalTrajs temporal_trajs, STLinks links, MetaFramesV3 frames_v3,
            vector<T> parents, double c_max
            );
};

template<class T>
MetaSTs<T>::MetaSTs(
        TemporalTrajs temporal_trajs, STLinks links, MetaFramesV3 frames_v3,
        vector<T> parents, double c_max
        )
    : temporal_trajs_{temporal_trajs}, st_links_{links}, frames_v3_{frames_v3},
    parents_{parents}, labels_{}, c_max_{c_max}, size_{0} {
        int v = 0;
        for (auto frames : frames_v3_){  ///< iter over views  -> vector< Coord2D >
            int f = 0;
            for (auto frame : frames){  ///< iter over frames -> Coord2D
                labels_[v].push_back(set<int>{});
                for (int  i = 0; i < frame.size(); i++){ ///< particle IDs
                    labels_[v][f].insert(i);
                }
                f++;
            }
            v++;
        }
        near_root_ = bool(is_same<T, StereoTrajs>::value);
}

template<class T>
void MetaSTs<T>::get_validate_trajs(){
    unsigned long total_frames = get_total_frames();
    for (int k1 = 0; k1 < temporal_trajs_[0].size(); k1++){
    for (int k2 = 0; k2 < temporal_trajs_[1].size(); k2++){
    for (int k3 = 0; k3 < temporal_trajs_[2].size(); k3++){
        MetaST<T> st{
            k1, k2, k3, c_max_, total_frames,
            frames_v3_, parents_, temporal_trajs_, st_links_
        };
        if (st.is_valid_){
            trajs_.push_back(st);
            size_++;
        }
    }}}
}

template<class T>
void MetaSTs<T>::add(MetaST<T> traj){
    if (traj.is_valid_){
        trajs_.push_back(traj);
        size_++;
    }
}

template<class T>
void MetaSTs<T>::clear(){
    trajs_.clear();
    size_ = 0;
}

template<class T>
unsigned long MetaSTs<T>::get_total_frames(){
    unsigned long frame_num = frames_v3_[0].size();
    parents_[0].get_total_frames(frame_num);
    return frame_num;
}

template<class T>
void MetaSTs<T>::get_total_frames(unsigned long& frame_num){
    frame_num *= frames_v3_[0].size();
    parents_[0].get_total_frames(frame_num);
}

template<class T>
vector<st::Coord3D> MetaSTs<T>::get_coordinates(array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os){
    vector<st::Coord3D> result;
    for (auto traj : trajs_){
        result.push_back(traj.get_coordinates(Ps, Os));
    }
    return result;
}

/**
 * Generating variables for optimisaing the stereo-linked 2D trajectories
 *
 * @param env:    the environment required by CPLEX
 * @param system: a collection of possible stereo-linked 2D trajectories
 *
 * @return: a 1D variable array, x[i] correspoinds to system.links_[i]
 */
template<class T>
IloBoolVarArray get_variables(IloEnv& env, T system){
    IloBoolVarArray x(env);
    for (int i = 0; i < system.size_; i++){
        x.add(IloBoolVar(env));
    }
    return x;
}

/**
 * Generating the "confined" constrains that essentially mean
 *     1. Any feature in any view must be passed by at least one trajectory
 * For a mathematical description, see eq. (3) in: 10.1109/TPAMI.2015.2414427
 */
template<class T>
IloRangeArray get_constrains_confined(IloEnv& env, IloBoolVarArray& x, T system){
    IloRangeArray constrains(env);
    IloInt idx;
    int total_trajs = 0;  ///< number of stereo trajs passing a 2d feature
    for (int view = 0; view < 3; view++) { ///< ∀ i
        for (int frame = 0; frame < system.labels_[view].size(); frame++) { 
            for (auto particle_id : system.labels_[view][frame]){
                IloExpr sum(env);
                idx = 0;
                total_trajs = 0;
                for (auto traj : system.trajs_){
                    if (traj.contains_particle(view, frame, particle_id)){
                        sum += x[idx];  ///< ∑(j)[ x(ij) ]
                        total_trajs++;
                    }
                    idx++;
                }
                if (total_trajs > 0){
                    constrains.add(sum >= 1);   ///< ≥ 1
                }
            }
        }
    }
    return constrains;
}

/**
 * Generating the "confined" cost function for optimising the temporal linking
 * The cost function is ∑(γ)[error(γ) * x(γ)]
 */
template<class T>
IloExpr get_cost_confined(IloEnv& env, IloBoolVarArray x, T system){
    IloExpr cost(env);
    for (int i = 0; i < system.size_; i++){
        cost += system.trajs_[i].error_ * x[IloInt(i)];
    }
    return cost;
}

/**
 * Optimise a stereo linked 2d trajectories using linear programming
 *
 * @param sys: a collection of stereo-lined 2d trajectories
 * @return: a collection of valid stereo links 
 */
template<class T>
T optimise_links_confined(T system){
    T new_system{system};
    new_system.clear();
    IloEnv   env;
    IloModel model(env);

    IloBoolVarArray x = get_variables(env, system);  // variables
    IloRangeArray constrains = get_constrains_confined(env, x, system);
    IloExpr cost = get_cost_confined(env, x, system);

    model.add(IloMinimize(env, cost));
    model.add(constrains);

    IloCplex cplex(model);
    cplex.setOut(env.getNullStream());  ///< avoid log msgs on the screen

    if ( !cplex.solve() ) {
        env.error() << "Failed to optimize LP" << endl;
        throw(-1);
    }

    IloNumArray vals(env);
    cplex.getValues(vals, x);
    for (int i = 0; i < system.size_; i++){
        if (vals[i] == 1){
            new_system.add(system.trajs_[i]);
        }
    }
    env.end();
    return new_system;
}

/**
 * Extract stereo links from many lower-level meta-stereo trajectories
 *     for a higher-level meta-stereo trajectory
 * @param systems: a collection of stereo links or meta stereo links
 *                 they are ordered in the time-axis
 */
template<class T>
STLinks get_meta_stereo_links(vector<T> systems){
    STLinks stereo_links;
    for (int frame = 0; frame < systems.size(); frame++){
        st::Links tmp{};
        int idx = 0;
        for (auto traj : systems[frame].trajs_){
            //tmp.add(traj.id_[0], traj.id_[1], traj.id_[2], traj.error_);
            tmp.add(idx, idx, idx, traj.error_);
            idx++;
        }
        stereo_links.push_back(tmp);
    }
    return stereo_links;
}

/**
 * Extract meta frames from many lower-level meta-stereo trajectories
 *     for a higher-level meta-stereo trajectory
 * @param systems: a collection of stereo links or meta stereo links
 *                 they are ordered in the time-axis
 */
template<class T>
MetaFramesV3 get_meta_frames(vector<T> systems){
    MetaFramesV3 frames_v3;
    int frame = 0;
    for (auto system : systems){
        for (int view = 0; view < 3; view++){
            frames_v3[view].push_back(tp::MetaFrame{});
        }
        for (auto traj : system.trajs_){
            for (int view = 0; view < 3; view++){
                tp::MetaParticle mp {traj.pos_start_[view], traj.pos_predict_[view]};
                frames_v3[view][frame].push_back(mp);
            }
        }
        frame++;
    }
    return frames_v3;
}


#endif
