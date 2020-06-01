#include "greta.h"


bool StereoTraj::contains_particle(int view, int frame, int id){
    return labels_[view][frame] == id;
}

double StereoTraj::error(){
    double sum = 0;
    double total = 0;
    for (auto e : st_errors_){
        sum += e;
        total++;
    }
    return sum / total; 
}

st::Coord3D StereoTraj::get_coordinates(FramesV3& frames_v3,
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
        ){
    int n_frames = frames_v3[0].size();
    st::Coord3D result{n_frames, 3};
    array<st::Vec2D, 3> coordinates_2d;
    for (int t = 0; t < n_frames; t++){
        for (int view = 0; view < 3; view++){
            coordinates_2d[view] = frames_v3[view][t].row(labels_[view][t]);
        }
        result.row(t) = st::three_view_reconstruct(coordinates_2d, Ps, Os);
    }
    return result;
}

StereoTraj::StereoTraj(
        int k1, int k2, int k3,
        double c_max, unsigned long frame_num,
        const Trajs2DV3& trajs_2d_v3, const STLinks& links)
    : id_{k1, k2, k3}, is_valid_{false}, c_max_{c_max},
      labels_{}, st_indices_{}, st_errors_{},
      trajs_2d_{ trajs_2d_v3[0][k1], trajs_2d_v3[1][k2], trajs_2d_v3[2][k3]} {
        array<int, 3> indices_v3;
        int st_index = 0;
        for (int frame = 0; frame < frame_num; frame++){
            bool link_found = false;
            for (int view = 0; view < 3; view++ ){
                indices_v3[view] = trajs_2d_[view][frame];
                labels_[view].push_back(trajs_2d_[view][frame]);
            }
            st_index = 0;
            // find stereo link that link 2d trajectories in current frame
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


StereoTrajs::StereoTrajs(
        Trajs2DV3& trajs_2d_v3, STLinks& links, FramesV3& frames_v3, double c_max
        )
    : trajs_2d_v3_{trajs_2d_v3}, st_links_{links}, frames_v3_{frames_v3},
      c_max_{c_max}, labels_{}, size_{0}, frame_num_{frames_v3[0].size()}{
        int v = 0;
        for (auto frames : frames_v3_){  ///< iter over views  -> vector< Coord2D >
            int f = 0;
            for (auto frame : frames){  ///< iter over frames -> Coord2D
                labels_[v].push_back(set<int>{});
                for (int  i = 0; i < frame.rows(); i++){ ///< particle IDs
                    labels_[v][f].insert(i);
                }
                f++;
            }
            v++;
        }
}

vector<st::Coord3D> StereoTrajs::get_coordinates(
        array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
        ){
    vector<st::Coord3D> result;
    for (auto traj : trajs_){
        result.push_back(traj.get_coordinates(frames_v3_, Ps, Os));
    }
    return result;
}

void StereoTrajs::get_validate_trajs(){
        for (int k1 = 0; k1 < trajs_2d_v3_[0].size(); k1++){
        for (int k2 = 0; k2 < trajs_2d_v3_[1].size(); k2++){
        for (int k3 = 0; k3 < trajs_2d_v3_[2].size(); k3++){
                StereoTraj st{k1, k2, k3, c_max_, frame_num_, trajs_2d_v3_, st_links_};
                if (st.is_valid_){
                    trajs_.push_back(st);
                    size_++;
                }
        }}}
}

void StereoTrajs::add(StereoTraj traj){
    if (traj.is_valid_){
        trajs_.push_back(traj);
        size_++;
    }
}


IloBoolVarArray get_variables(IloEnv& env, StereoTrajs system){
    IloBoolVarArray x(env);
    for (int i = 0; i < system.size_; i++){
        x.add(IloBoolVar(env));
    }
    return x;
}


IloRangeArray get_constrains_confined(IloEnv& env, IloBoolVarArray& x, StereoTrajs system){
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


IloExpr get_cost_confined(IloEnv& env, IloBoolVarArray x, StereoTrajs system){
    IloExpr cost(env);
    for (int i = 0; i < system.size_; i++){
        cost += system.trajs_[i].error() * x[IloInt(i)];
    }
    return cost;
}


StereoTrajs optimise_links_confined(StereoTrajs system){
    StereoTrajs new_system{
        system.trajs_2d_v3_, system.st_links_, system.frames_v3_, system.c_max_
    };
    IloEnv   env;
    IloModel model(env);

    IloBoolVarArray x = get_variables(env, system);  // variables
    IloRangeArray constrains = get_constrains_confined(env, x, system);
    IloExpr cost = get_cost_confined(env, x, system);

    model.add(IloMinimize(env, cost));
    model.add(constrains);

    IloCplex cplex(model);
    //cplex.setOut(env.getNullStream());  ///< avoid log msgs on the screen

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
