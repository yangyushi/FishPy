#ifndef TEMPORAL
#define TEMPORAL
#include <ilcplex/ilocplex.h>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <set>
#include <map>

ILOSTLBEGIN  ///< marco from CPLEX for cross-platform compability
using namespace std;

namespace temporal{

using Coord2D = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Vec2D = Eigen::Matrix<double , 2, 1>;
using PYLinks = vector< array<int, 2> >;
using Label = Eigen::Array<int, Eigen::Dynamic, 1>;
using Labels = vector<Label>;
/**
 * shape: (2, frames), axis 0 is (indices, time_points)
 */
using PYTraj = array<vector<int>, 2>;
using PYTrajs = vector<PYTraj>;

/**
 * A meta particle that contains its intial 2D position 
 *     and the predicted position at its next frame
 */
using MetaParticle = array<Vec2D, 2>;    ///< shape (2, )
using MetaFrame = vector<MetaParticle>;  ///< shape (n_particle, 2)
using MetaFrames = vector<MetaFrame>;    ///< shape (n_frames, n_particles, 2)

/**
 * temporally matched indices
 */
struct Link{
    array<int, 2> indices_; ///< the feature IDs in 2 frames
    double error_; ///< the distance between prediction and result
    string repr_;
    Link(int i, int j, double e);
    int& operator[] (int index);
};

/**
 * A collection of temporal links for constrained optimisation
 * These links were subjected to a spatial cutoff of the distance
 *     between prediction and result, and they form the set T in
 *     equation (3) of the supplementary info of
 *     this paper: 10.1109/TPAMI.2015.2414427
 */
struct Links{
    vector<Link> links_;
    int size_;
    array<set<int>, 2> indices_;  ///< particle IDs in 2 frames, the indices of an (imaginary) 2D matrix
    void report();
    void add(int i, int j, double d);
    void add(Link l);
    PYLinks to_py();
    Link& operator[] (int index);
    Links();
    Links(vector<Link> links);
};


/*
 * a 2d path represented by the indices of the same particle, "shape": (nframe, 2)
 * in different frames, the γ_i(t) in 10.1109/TPAMI.2015.2414427
 * traj[t] --> particle id in frame t
 */
struct Traj{
    vector<int> indices_;
    vector<int> time_;
    Traj();
    Traj(int idx, int t);
    Traj(array<int, 2> indices, int t0, int t1);
    Traj(int i0, int i1, int t0, int t1);
    Traj(vector<int> indices, vector<int> time_points);
    Traj(const Traj& t);
    void add(int idx, int t);
    int& operator[] (int index);
    int last_time();
};

/**
 * shape (n_trajs, n_frames, 2)
 */
using Trajs = vector<Traj>;

/**
 * Implementation of nearest neighbour linker
 * see 10.1007/s00348-005-0068-7 for detail
 * This is used as base for other more advanced linkers
 */
struct LinkerNN{
    double sr_; 
    /**
     * Generating all possible linke between two successive frames
     * @param: f0 - xy coordinates of current frame
     * @param: f1 - xy coordinates of next frame
     */
    Links get_links(Coord2D f0, Coord2D f1);
    Links get_links(MetaFrame f0, MetaFrame f1);
    LinkerNN(double search_range);
};

/**
 * Implementation of nearest neighbour linker
 * see 10.1007/s00348-005-0068-7 for detail
 * This is used as base for other more advanced linkers
 */
struct LinkerF3 : public LinkerNN{
    /**
     * Calculation the distance between prediction and
     *     the actual position in the next frame based
     *     on the 3 frame minimum acceleration heuristic
     */
    Vec2D get_distance_f3(Vec2D x0, Vec2D x1, Vec2D xp);
    /**
     * Generating all possible linke between two successive frames
     * @param: f0 - xy coordinates of current frame
     * @param: f1 - xy coordinates of next frame
     * @param: fp - xy coordinates of previous frame
     * @param: links_lp0 - the links between the previouse frame and the current frame
     */
    using LinkerNN::get_links;
    Links get_links(Coord2D& f0, Coord2D& f1, Coord2D& fp, const Links& links_p0);
    LinkerF3(double search_range);
};

/**
 * search all the positions in frame 1, if its distance to x0
 *     is smaller than sr, add link i, j, distance
 *     the distance is calculated as |x0 - x1|
 */
void collect_link(Vec2D x0, Coord2D& f1, Links& links, int i, double sr);

/**
 * search all the positions in frame 1, if its distance to x0
 *     is smaller than sr, add link i, j, distance
 *     the distance is calculated as |xp + x1 - 2 * x1|
 */
void collect_link(Vec2D xp, Vec2D x0, Coord2D& f1, Links& links, int i, double sr);

/**
 * search all the meta particles in frame 1, if its distance to the prediction of x0
 *     is smaller than sr, add link i, j, distance
 *     the distance is calculated as |x0.prediction - x1.start|
 */
void collect_link(MetaParticle x0, MetaFrame& f1, Links& links, int i, double sr);

/**
 * Generating variables for optimisaing the temporal linking result
 *
 * @param env: the environment required by CPLEX
 * @param sys: a collection of possible 2-frame temporal links
 *
 * @return: a 1D variable array, x[i] correspoinds to system.links_[i]
 */
IloBoolVarArray get_variables(IloEnv& env, Links system);

/**
 * Generating the constrains that essentially mean
 *     1. Any feature in frame i must have its correspondances in frame i+1
 *     2. The same feature i can be used for different termporal links
 * For a mathematical description, see eq. (3) in the supplementary info of
 *     this paper: 10.1109/TPAMI.2015.2414427
 */
IloRangeArray get_constrains(IloEnv& env, IloBoolVarArray& x, Links system);

/**
 * Generating the cost function for optimising the temporal linking
 * The cost function is ∑(ij)[error(ij) * x(ij)]
 */
IloExpr get_cost(IloEnv& env, IloBoolVarArray x, Links system);

/**
 * Optimise a stereo link result using linear programming
 *
 * @param sys: a collection of possible 3 view stereo links
 * @return: a collection of valid stereo links 
 */
Links optimise_links(Links system);

/**
 * Link positions from 2 successive frames
 */
Trajs link_2d(vector<Coord2D> frames, double search_range, bool allow_fragment);

/**
 * Link meta particles from 2 successive "frames"
 */
Trajs link_meta(MetaFrames frames, double search_range, bool allow_fragment);

Trajs links_to_trajs(vector<Links> links, bool allow_fragment);

Labels links_to_labels(vector<Links> links, const vector<Coord2D>& frames);

}
#endif
