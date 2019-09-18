#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include "voro++.hh"

#ifndef _VORO_HPP
#define _VORO_HPP

using namespace std;
using namespace voro;


array<double, 3> get_projection(double x, double y, double z, double c);

vector<double> get_volumes(container &box);


class Wall_tank : public wall {
    /*
     * The curvy surface of the tank for voro++ to cut voronoi cell.
     */
    public:
        Wall_tank(double coefficient, int wall_id);

        bool point_inside(double x, double y, double z);

        template<class vc_class>
        inline bool cut_cell_base(vc_class &c, double x, double y, double z){
            /*
             * x, y, z -> location of the Voronoi cell
             * 1. Find the projection of (xyz) to the surface
             * 2. Find the shift vector from (xyz) to projection, double it
             * 3. Construct `nplane(double_shift, w_id)` to cut the cell
             */
            array<double, 3> proj = get_projection(x, y, z, this->coef);
            double shift_x = 2 * (proj[0] - x);
            double shift_y = 2 * (proj[1] - y);
            double shift_z = 2 * (proj[2] - z);
            return c.nplane(shift_x, shift_y, shift_z, this->w_id);
        }

        bool cut_cell(voronoicell          &c, double x, double y, double z) {return cut_cell_base(c, x, y, z);}
        bool cut_cell(voronoicell_neighbor &c, double x, double y, double z) {return cut_cell_base(c, x, y, z);}

    private:
        const double coef;
        const int w_id;
};


class Fish_tank {
    /*
     * The container enclosing the fish
     * It is composed of the curvy surface and the top water level (fish don't swim above water)
     */
    public:

        Wall_tank  wall_side;
        container  core;

        Fish_tank(double z, double c);

        bool is_inside(double x, double y, double z);

        void put(int index, double x, double y, double z);

        double get_volume();

        vector<double> volumes();

    private:
        double z_max;
        double coef;
};


#endif
