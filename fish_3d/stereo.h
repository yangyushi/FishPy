#ifndef STEREO
#define STEREO
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <cmath>

using namespace std;

using Mat33   = Eigen::Matrix<double, 3,              3, Eigen::RowMajor>;
using ProjMat = Eigen::Matrix<double, 3,              4, Eigen::RowMajor>;
using Coord2D = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Line    = Eigen::Matrix<double, 2,              3, Eigen::RowMajor>;
using Lines = array<Line, 3>;

using Vec2D   = Eigen::Matrix<double, 2, 1>;
using Vec3D   = Eigen::Matrix<double, 3, 1>;
using Vec3DH  = Eigen::Matrix<double, 4, 1>;

using Link = array<int, 3>;  // stereoly matched indices
using Links = vector<Link>;

using TriXY = array<Vec2D, 3>;
using TriXYZ = array<Vec3D, 3>;
using TriPM = array<ProjMat, 3>;


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

Vec2D reproject_refractive(Vec3D point, Vec2D xy, ProjMat P);

double get_reproj_error(Vec3D xyz, TriXY centres, TriPM Ps, TriXYZ Os);

double get_error(TriXY centres, TriPM Ps, TriXYZ Os);

Links three_view_match(
        Coord2D& centres_1, Coord2D& centres_2, Coord2D& centres_3,
        ProjMat P1, ProjMat P2, ProjMat P3,
        Vec3D O1, Vec3D O2, Vec3D O3, double tol_2d
        );

#endif

