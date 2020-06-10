#include "temporal.h"

namespace temporal {

Link::Link(int i, int j, double e)
    : indices_{i, j}, error_{e}{
        ostringstream r;
        r   << "[" << i << ", " << j << "], error: "
            << fixed << setprecision(2) << e;
        repr_ = r.str();
}


int& Link::operator[] (int index){
    return indices_[index];
}

Links::Links() : links_{}, size_{0} {}

Links::Links(vector<Link> links)
    : links_{links}, size_{int(links.size())} {
    for (auto link : links_){
        for (int i = 0; i < 2; i++){
            indices_[i].insert(link[i]);
        }
    }
}

Link& Links::operator[] (int index){
    return links_[index];
}

void Links::report(){
    for (auto l : links_){
        cout << l.repr_ << endl;
    }

    for (int i = 0; i < 2; i++){
        cout << "ID " << i+1 << ": ";
        int count = 0;
        for (auto idx : indices_[i]){
            if (++count < indices_[i].size()) {
                cout << idx << ", ";
            } else {
                cout << idx << endl;
            }
        }
    }
}

void Links::add(int i, int j, double d){
    bool must_be_new =
        indices_[0].find(i) == indices_[0].end() or
        indices_[1].find(j) == indices_[1].end();
    if (must_be_new){
        links_.push_back(Link{i, j, d});
        indices_[0].insert(i);
        indices_[1].insert(j);
        size_ ++;
        return;
    } else {
        for (auto& l0 : links_){
            if ((l0[0] == i) and (l0[1] == j)){
                if (d < l0.error_){
                    l0.error_ = d;
                    ostringstream r;
                    r   << "[" << i << ", " << j << "], error: "
                        << fixed << setprecision(2) << d;
                    l0.repr_ = r.str();
                }
                return;
            }
        }
        links_.push_back(Link{i, j, d});
        indices_[0].insert(i);
        indices_[1].insert(j);
        size_ ++;
        return;
    }
}


void Links::add(Link l){
    bool must_be_new =
        indices_[0].find(l[0]) == indices_[0].end() or
        indices_[1].find(l[1]) == indices_[1].end();
    if (must_be_new){
        links_.push_back(l);
        indices_[0].insert(l[0]);
        indices_[1].insert(l[1]);
        size_++;
        return;
    } else {
        for (auto& l0 : links_){
            if ((l0[0] == l[0]) and (l0[1] == l[1])){
                if (l.error_ < l0.error_){
                    l0.error_ = l.error_;
                    ostringstream r;
                    r   << "[" << l[0] << ", " << l[1] << "], error: "
                        << fixed << setprecision(2) << l.error_;
                    l0.repr_ = r.str();
                }
                return;
            }
        }
        links_.push_back(l);
        indices_[0].insert(l[0]);
        indices_[1].insert(l[1]);
        size_++;
        return;
    }
}


PYLinks Links::to_py(){
    PYLinks result;
    for (auto link : links_){
        result.push_back(link.indices_);
    }
    return result;
}


Traj::Traj()
    : indices_{}, time_{} {}

Traj::Traj(int idx, int t)
    : indices_{idx}, time_{t} {}

Traj::Traj(array<int, 2> indices, int t0, int t1)
    : indices_{indices[0], indices[1]}, time_{t0, t1} {}

Traj::Traj(int i0, int i1, int t0, int t1)
    : indices_{i0, i1}, time_{t0, t1} {}

Traj::Traj(vector<int> indices, vector<int> time_points)
    : indices_{indices}, time_{time_points} {}

Traj::Traj(const Traj& t)
    : indices_{t.indices_}, time_{t.time_} {}

void Traj::add(int idx, int t){
    indices_.push_back(idx);
    time_.push_back(t);
}

int& Traj::operator[] (int index) {
    if (index < 0){
        return indices_[indices_.size() + index];   ///< a[-1] -> last element
    } else {
        return indices_[index];
    }
}

int Traj::last_time() { return time_[time_.size() - 1]; }


LinkerNN::LinkerNN(double search_range) : sr_{search_range} {}


Links LinkerNN::get_links(Coord2D f0, Coord2D f1){
    Links result;
    Vec2D x0;
    for (int i=0; i < f0.rows(); i++){
        x0 = f0.row(i);
        collect_link(x0, f1, result, i, sr_);
    }
    return result;
}

Links LinkerNN::get_links(MetaFrame f0, MetaFrame f1){
    Links result;
    MetaParticle x0;
    for (int i=0; i < f0.size(); i++){
        x0 = f0[i];
        collect_link(x0, f1, result, i, sr_);
    }
    return result;
}


LinkerF3::LinkerF3(double search_range) : LinkerNN{search_range} {}


Links LinkerF3::get_links(Coord2D& f0, Coord2D& f1, Coord2D& fp, const Links& links_p0){
    Links result;
    set<int> indices_0 = links_p0.indices_[1];  ///< the indices of f0 linking to fp
    Vec2D x0, xp;

    for (int i=0; i < f0.rows(); i++){
        if (indices_0.find(i) == indices_0.end()){  ///< index i is NOT linked to fp
            x0 = f0.row(i);
            collect_link(x0, f1, result, i, sr_);
        } else {    ///< index i is linked to fp
            for (auto lp : links_p0.links_){
                if (lp[1] == i){
                    xp = fp.row(lp[0]); 
                    x0 = f0.row(lp[1]);
                    collect_link(xp, x0, f1, result, i, sr_);
                }
            }
        }
    }
    return result;
}


void collect_link(Vec2D x0, Coord2D& f1, Links& links, int i, double sr){
    Vec2D x1;
    bool link_found{false};
    int nn_idx{0};
    double nn_dist = (x0 - Vec2D{f1.row(0)}).norm();
    for (int j=0; j < f1.rows(); j++){
        x1 = f1.row(j);
        double dist = (x0 - x1).norm();
        if (dist < nn_dist) {
            nn_dist = dist;
            nn_idx = j;
        }
        if (dist <= sr){
            links.add(i, j, dist);
            link_found = true;
        }
    }
    if (not link_found){
        links.add(i, nn_idx, nn_dist);
    }
}


void collect_link(MetaParticle x0, MetaFrame& f1, Links& links, int i, double sr){
    MetaParticle x1;
    bool link_found{false};
    int nn_idx{0};
    double nn_dist = (x0[1] - f1[0][0]).norm();
    for (int j=0; j < f1.size(); j++){
        x1 = f1[j];
        double dist = (x0[1] - x1[0]).norm();  ///< between prediction and observation
        if (dist < nn_dist) {
            nn_dist = dist;
            nn_idx = j;
        }
        if (dist <= sr){
            links.add(i, j, dist);
            link_found = true;
        }
    }
    if (not link_found){
        links.add(i, nn_idx, nn_dist);
    }
}

void collect_link(Vec2D xp, Vec2D x0, Coord2D& f1, Links& links, int i, double sr){
    Vec2D x1;
    bool link_found{false};
    int nn_idx{0};
    double nn_dist = (x0 - Vec2D{f1.row(0)}).norm();
    for (int j=0; j < f1.rows(); j++){
        x1 = f1.row(j);
        double dist = (xp + x1 - 2 * x0).norm();
        if (dist < nn_dist) {
            nn_dist = dist;
            nn_idx = j;
        }
        if (dist <= sr){
            links.add(i, j, dist);
        }
    }
    if (not link_found){
        links.add(i, nn_idx, nn_dist);
    }
}


IloBoolVarArray get_variables(IloEnv& env, Links system){
    IloBoolVarArray x(env);
    for (int i = 0; i < system.size_; i++){
        x.add(IloBoolVar(env));
    }
    return x;
}


IloRangeArray get_constrains(IloEnv& env, IloBoolVarArray& x, Links system){
    IloRangeArray constrains(env);
    IloInt idx;
    for (int frame = 0; frame < 2; frame++) { 
        for (auto i : system.indices_[frame]) { ///< ∀ i
            IloExpr sum(env);
            idx = 0;
            for (auto link : system.links_){
                if (link[frame] == i){
                    sum += x[idx];  ///< ∑(j)[ x(ij) ]
                }
                idx++;
            }
            constrains.add(sum >= 1);   ///< ≥ 1
        }
    }
    return constrains;
}


IloExpr get_cost(IloEnv& env, IloBoolVarArray x, Links system){
    IloExpr cost(env);
    for (int i = 0; i < system.size_; i++){
        cost += system.links_[i].error_ * x[IloInt(i)];
    }
    return cost;
}


Links optimise_links(Links system){
    Links new_system;
    IloEnv   env;
    IloModel model(env);

    IloBoolVarArray x = get_variables(env, system);  // variables
    IloRangeArray constrains = get_constrains(env, x, system);
    IloExpr cost = get_cost(env, x, system);

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
            new_system.add(system.links_[i]);
        }
    }
    env.end();
    return new_system;
}


void extend_trajectories(int frame, Trajs& trajs, Links& links_01){
    Trajs new_trajs{};
    for (auto& traj : trajs){
        if (traj.last_time() != frame){ continue; }
        Traj traj_copy{traj};
        bool should_branch = false;
        for (auto link : links_01.links_){
            if (link[0] == traj_copy[-1]){
                if (not should_branch) {  ///< extending a trajectory
                    traj.add(link[1], frame + 1);
                    should_branch = true;
                } else {  ///< branching & extend a trajectory
                    Traj new_traj{traj_copy};
                    new_traj.add(link[1], frame + 1);
                    new_trajs.push_back(new_traj);
                }
            }
        }
    }
    trajs.insert(  ///< concatenating trajs and new_trajs
        trajs.end(), new_trajs.begin(), new_trajs.end()
    );
}


void add_new_trajectories(int frame, Trajs& trajs, Links& links_01, Links& links_p0){
    Trajs new_trajs{};
    vector<int> new_indices;
    for (int i : links_01.indices_[0]){
        if ( // if i is not in second indices of links_p0
            links_p0.indices_[1].find(i) == links_p0.indices_[1].end()
            ){
            new_indices.push_back(i);
        }
    }
    for (int i : new_indices){
        for (auto link : links_01.links_){
            if (link[0] == i){
                new_trajs.push_back(Traj{link.indices_, frame, frame + 1});
            }
        }
    }
    trajs.insert(  ///< concatenating trajs and new_trajs
        trajs.end(), new_trajs.begin(), new_trajs.end()
    );
}


Trajs links_to_trajs(vector<Links> links_multi_frames, bool allow_fragment){
    Trajs trajs;
    // adding links from first frame
    for (auto link : links_multi_frames[0].links_){
        trajs.push_back( Traj{link.indices_, 0, 1} );
    }
    // adding links from the 2nd frame to second last frame
    for (int f = 1; f < links_multi_frames.size(); f++){
        extend_trajectories(f, trajs, links_multi_frames[f]);
        if (allow_fragment){
            add_new_trajectories(f, trajs, links_multi_frames[f], links_multi_frames[f-1]);
        }
    }
    if (not allow_fragment){  // remove trajectories shorter than the frame number
        int frame_num = links_multi_frames.size() + 1;
        Trajs trajs_full;
        for (auto traj : trajs){
            if (traj.indices_.size() == frame_num){
                trajs_full.push_back(traj);
            }
        }
        return trajs_full;
    };
    return trajs;
}


Trajs link_2d(vector<Coord2D> frames, double search_range, bool allow_fragment){
    vector<Links> links_multi_frames;
    LinkerF3 linker{search_range};
    Links links;
    for (int f = 0; f < frames.size() - 1; f++){
        if (f == 0){
            links = linker.get_links(frames[f], frames[f + 1]);
        } else {
            links = linker.get_links(
                        frames[f], frames[f + 1], frames[f - 1],
                        links_multi_frames[links_multi_frames.size() - 1]
                    );
        }
        links = optimise_links(links);
        links_multi_frames.push_back(links);
    }
    return links_to_trajs(links_multi_frames, allow_fragment);
}


Trajs link_meta(MetaFrames frames, double search_range, bool allow_fragment){
    vector<Links> links_multi_frames;
    LinkerNN linker{search_range};
    Links links;
    for (int f = 0; f < frames.size() - 1; f++){
        links = linker.get_links(frames[f], frames[f + 1]);
        links = optimise_links(links);
        links_multi_frames.push_back(links);
    }
    return links_to_trajs(links_multi_frames, allow_fragment);
}


Labels links_to_labels(vector<Links> links, const vector<Coord2D>& frames){
    Labels labels;
    set<int> label_set;

    labels.push_back( Label::LinSpaced(1, 0, frames[0].rows()) );
    for (int idx = 0; idx < frames[0].rows(); idx++){
        label_set.insert(idx);
    }

    for (int frame = 0; frame < links.size(); frame++) {
        Label label_fp = labels[labels.size() - 1];
        Label label_f0{frames[frame + 1].rows(), 1};
        Links links_p0 = links[frame];  // previous to current link
        for (auto l : links_p0.links_){
            label_f0.row(l[1]) = label_fp.row(l[0]);
        }
        for (int i = 0; i < label_f0.rows(); i++){
            // if index i is not found in links_p0
            if (links_p0.indices_[1].count(i) == links_p0.indices_[1].size()){
                int new_label = *label_set.rbegin() + 1;
                label_f0.row(i) = new_label;
                label_set.insert(new_label);
            }
        }
        labels.push_back(label_f0);
    }
    return labels;
}

}
