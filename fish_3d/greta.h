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
 */
using Traj2DV3  = array<tp::Traj, 3>;

/*
 * 2D Trajectories in different views
 *    "shape" (3, n_frame, 2)
 *    the last axis is (indices, time)
 */
using Trajs2DV3 =  array<tp::Trajs, 3>;

using STIndices = vector<int>;

using Traj2DV3s = vector<Traj2DV3>;   ///< shape (n_paths, 3, n_frame, 2)

/**
 * stereo links in different frames, shape (n_frame, n_links, )
 */
using STLinks = vector<st::Links>;


/**
 * γ(t)
 * k1, k2, k3 are indices of 2d trajectories
 */
struct StereoTraj{
    bool is_valid_;  ///< false if no stereo link in any frame
    double c_max_;
    vector<int> st_indices_;  ///< the ID of one stereo link, -1 means nan. shape (n_frame,)
    /**
     * The reprojection error in each frame, shape (frame, )
     * if there is no stereo link in frame i, st_indices_[f] == c_max_
     */
    vector<double> st_errors_;
    Traj2DV3 trajs_2d_; ///< 3 temporal trajectories, shape (3, frame)
    StereoTraj(
            int k1, int k2, int k3,
            double c_max, int frames,
            const Trajs2DV3& trajs_2d_v3,
            const STLinks& links
            );
};


/**
 * { γ(t) }
 */
struct StereoTrajs{
    double c_max_;
    int frames_;
    vector<double> error_;
    vector<StereoTraj> trajs_;
    StereoTrajs(Trajs2DV3 trajs_2d_v3, STLinks links, double c_max, int frames);
};


#endif
