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
 * 2D Trajectories in different views
 *    "shape" (3, n_frame, 2)
 *    the last axis is (indices, time)
 *    tp::traj[t] --> particle id in frame t
 */
using Traj2DV3  = array<tp::Traj, 3>;

/*
 * 2D Trajectories in different views
 *    "shape" (3, n_trajs, n_frame, 2)
 *    the last axis is (indices, time)
 */
using Trajs2DV3 =  array<tp::Trajs, 3>;

using STIndices = vector<int>;

using Labels = vector< set<int> >;    ///< shape (n_frames, n_particles)
using Frames = vector< tp::Coord2D >; ///< shape (n_frames, n_particles, 2)
using LabelsV3 = array<Labels  , 3>;  ///< shape (3, n_frames, n_particles)
using FramesV3 = array<Frames, 3>;    ///< shape (3, n_frames, n_particles, 2)

/**
 * stereo links in different frames, shape (n_frame, n_links, )
 */
using STLinks = vector<st::Links>;


/**
 * γ(t)
 * k1, k2, k3 are indices of 2d trajectories
 */
struct StereoTraj{
    array<int, 3> id_;
    bool is_valid_;  ///< false if no stereo link in any frame
    double c_max_;
    array<vector<int>, 3> labels_;
    /**
     * the ID of one stereo link, -1 means nan. shape (n_frame,)
     */
    vector<int> st_indices_;

    /**
     * The reprojection error in each frame, shape (frame, )
     * if there is no stereo link in frame i, st_indices_[f] == c_max_
     */
    vector<double> st_errors_;
    Traj2DV3 trajs_2d_; ///< 3 temporal trajectories, shape (3, frame)

    /**
     * Check if particle id is contained in a cetrain frame inside a certain view
     */
    inline bool contains_particle(int view, int frame, int id);

    /**
     * Calculate the stereo error of this stereo-linked trajectory
     */
    double error();

    /**
     * Calculate the 3D coordinates of this trajectory
     * @param Ps: projection matrices of three cameras
     * @param Os: origins of three cameras
     */
    st::Coord3D get_coordinates(
            FramesV3& frames,
            array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
            );

    StereoTraj(
            int k1, int k2, int k3,
            double c_max, unsigned long frames,
            const Trajs2DV3& trajs_2d_v3,
            const STLinks& links
            );
};


/**
 * { γ(t) }
 * All the 2d trajectories should have the same length in time
 */
struct StereoTrajs{
    Trajs2DV3& trajs_2d_v3_;
    STLinks& st_links_;
    FramesV3& frames_v3_;
    double c_max_;
    LabelsV3 labels_;
    int size_; ///< number of stereo trajectories, trajs_.size() essentially
    unsigned long frame_num_;
    vector<StereoTraj> trajs_;
    /**
    * Valid stereo trajs were selected in the constructor 
    */
    vector<st::Coord3D> get_coordinates(
            array<st::ProjMat, 3> Ps, array<st::Vec3D, 3> Os
            );
    void get_validate_trajs();
    void add(StereoTraj traj);
    /**
     * return indices of 3D trajectories
     */
    StereoTrajs(
            Trajs2DV3& trajs_2d_v3, STLinks& links, FramesV3& frames,
            double c_max
            );
};


/**
 * Generating variables for optimisaing the stereo-linked 2D trajectories
 *
 * @param env:    the environment required by CPLEX
 * @param system: a collection of possible stereo-linked 2D trajectories
 *
 * @return: a 1D variable array, x[i] correspoinds to system.links_[i]
 */
IloBoolVarArray get_variables(IloEnv& env, StereoTrajs system);


/**
 * Generating the "confined" constrains that essentially mean
 *     1. Any feature in any view must be passed by at least one trajectory
 * For a mathematical description, see eq. (3) in: 10.1109/TPAMI.2015.2414427
 */
IloRangeArray get_constrains_confined(
        IloEnv& env, IloBoolVarArray& x, StereoTrajs system
        );


/**
 * Generating the "confined" cost function for optimising the temporal linking
 * The cost function is ∑(γ)[error(γ) * x(γ)]
 */
IloExpr get_cost_confined(IloEnv& env, IloBoolVarArray x, StereoTrajs system);


/**
 * Optimise a stereo linked 2d trajectories using linear programming
 *
 * @param sys: a collection of stereo-lined 2d trajectories
 * @return: a collection of valid stereo links 
 */
StereoTrajs optimise_links_confined(StereoTrajs system);


#endif
