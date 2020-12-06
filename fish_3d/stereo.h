#ifndef STEREO
#define STEREO
#include <complex>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <cmath>
#include <set>
#include <tuple>
#include <ilcplex/ilocplex.h>

ILOSTLBEGIN  ///< marco from CPLEX for cross-platform compability
using namespace std;

namespace stereo {
    using Mat33   = Eigen::Matrix<double, 3,              3, Eigen::RowMajor>;
    using ProjMat = Eigen::Matrix<double, 3,              4, Eigen::RowMajor>;
    using Coord2D = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
    using Coord3D = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using Line    = Eigen::Matrix<double, 2,              3, Eigen::RowMajor>;
    using Lines = array<Line, 3>;

    using Vec2D   = Eigen::Matrix<double, 2, 1>;
    using Vec3D   = Eigen::Matrix<double, 3, 1>;
    using Vec3DH  = Eigen::Matrix<double, 4, 1>;

    using TriXY = array<Vec2D, 3>;
    using TriXYZ = array<Vec3D, 3>;
    using TriPM = array<ProjMat, 3>;

    /**
     * Indices and stereo errors of a trjaectory
     * compatible with pybind11
     * shape (n_links, 4)
     */
    using PYLinks = vector< tuple<int, int, int, double> >;

    /**
     * A collection of 3 view stereo linking result
     */
    struct Link{
        array<int, 3> indices_; ///< the feature IDs in 3 views
        double error_; ///< the averaged reprojection error
        string repr_;
        Link(int i, int j, int k, double e);
        int& operator[] (int index);
    };


    /**
     * A collection of stereo links for constrained optimisation, "shape": (n_links, 3)
     * These links were subjected to a spatial cutoff of the reprojection error,
     *     and they form the set S in equation (1) of the supplementary info of
     *     this paper: 10.1109/TPAMI.2015.2414427
     */
    struct Links{
        vector<Link> links_;
        int size_;
        array<set<int>, 3> indices_;  ///< particle IDs in 3 views, the indices of an (imaginary) 3D tensor
        void report();
        void add(int i, int j, int k, double e);
        void add(Link l);
        /**
         * collect result in a std containers for pybind11 to export as python objects
        */
        PYLinks to_py();
        Links();
        Links(vector<Link> links);
        Links(PYLinks links_py);
        Link& operator[] (int index);
    };


    /*
     * calculate the point on the water-air interface from xy coordinate in the image
     * the interface is supposed to be at z=0, and parallel to the xy plane
     * @param xy: the position of a feature in the 2D image
     * @param P: the projection matrix of the camera that took the 2D image 
     */
    Vec3D get_poi(Vec2D& xy, ProjMat& p);


    /*
     * calculate the direction of the refraction 
     * @param incidence: the vector of the incident ray
     * @param refraction_index: the refraction index of the media
     */
    Vec3D get_refraction_ray(Vec3D incidence);


    /*
     * calculate the intersection of different lines
     * @param lines: the vector of the Line, each line is (2, 3) array.
     *               the first row is a point on the line
     *               the second row is the unit direction of the line
     */
    Vec3D get_intersection(Lines lines);


    /*
     * The meanings of variable names can be found in this paper 10.1109/CRV.2011.26
     */
    double get_u(double d, double x, double z);


    Vec2D reproject_refractive(Vec3D point, ProjMat P, Vec3D O);

    double get_reproj_error(Vec3D xyz, TriXY centres, TriPM Ps, TriXYZ Os);


    double get_error(TriXY& centres, TriPM& Ps, TriXYZ& Os);
    double get_error_with_xyz(TriXY& centres, TriPM& Ps, TriXYZ& Os, Vec3D& xyz);

    /**
     * Calculate the 3D coordinates from 3 stereo-matched 2D coordinates
     */
    Vec3D three_view_reconstruct(
            array<Vec2D, 3>Cs, array<ProjMat, 3> Ps, array<Vec3D, 3> Os
            );

    /**
     * Generating stereo matched indices
     */
    Links three_view_match(
            Coord2D& centres_1, Coord2D& centres_2, Coord2D& centres_3,
            ProjMat P1, ProjMat P2, ProjMat P3,
            Vec3D O1, Vec3D O2, Vec3D O3, double tol_2d, bool optimise
            );

    /**
     * Generating variables for optimisaing the stereo linking result
     *
     * @param env: the environment required by CPLEX
     * @param sys: a collection of possible 3-view stereo links
     *
     * @return: a 1D variable array, x[i] correspoinds to system.stereo_links_[i]
     */
    IloBoolVarArray get_variables(IloEnv& env, Links system);


    /**
     * Generating the constrains that essentially mean
     *     1. Any feature in view i must have its correspondances in view j and k
     *     2. The same feature i can be used for different stereo links
     * For a mathematical description, see eq. (2) in the supplementary info of
     *     this paper: 10.1109/TPAMI.2015.2414427
     */
    IloRangeArray get_constrains(IloEnv& env, IloBoolVarArray& x, Links system);


    /**
     * Generating the cost function for optimising the stereolinking
     * The cost function is ∑(ijk)[error(ijk) * x(ijk)]
     */
    IloExpr get_cost(IloEnv& env, IloBoolVarArray x, Links system);


    /**
     * Optimise a stereo link result using linear programming
     *
     * @param sys: a collection of possible 3 view stereo links
     * @return: a collection of valid stereo links 
     */
    Links optimise_links(Links system);
}

#endif
