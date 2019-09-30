#include <iostream>
#include "voro.hpp"

array<double, 3> get_projection(double x, double y, double z, double c) {
    /*
     project a 3D location to the surface of fish tank
     the surface is defined by z = c * R^2, and it is assumed to be cylindrically symmetric
     the expression for radius_proj is analytical result obtained from Mathematica
     */
    array<double, 3> proj;
    double p = pow(2., 1. / 3.);
    double radius = sqrt(x * x + y * y);
    double theta  = atan2(y, x);
    double term = 108 * pow(c, 4) * radius + sqrt(11664 * pow(c, 8) * pow(radius, 2) + 864 * pow(c, 6) * pow((1 - 2 * c * z), 3));
    term = pow(term, 1. / 3.);
    double radius_proj = -(p * (1 - 2 * c * z)) / term + term / (6 * p * c * c);

    proj[0] = radius_proj * cos(theta);
    proj[1] = radius_proj * sin(theta);
    proj[2] = c * pow(radius_proj, 2.);
    return proj;
}


Wall_tank::Wall_tank(double coefficient, int wall_id=-99): coef(coefficient), w_id(wall_id) {
}

bool Wall_tank::point_inside(double x, double y, double z) {
    return z > (this->coef * (x*x + y*y));
}


Fish_tank::Fish_tank(double z, double c):
    wall_side(c, 90),
    core(-sqrt(z/c), sqrt(z/c), -sqrt(z/c), sqrt(z/c), 0, z, 8, 8, 8, false, false, false, 8){
        this->z_max = z;
        this->coef  = c;
        this->core.add_wall(this->wall_side);
    }

bool Fish_tank::is_inside(double x, double y, double z){
    bool in_water = (z <= this->z_max) and (z > 0);
    bool in_wall = z > (this->coef * (x*x + y*y));
    return (in_wall && in_water);
}

void Fish_tank::put(int index, double x, double y, double z) {
    this->core.put(index, x, y, z);
}

double Fish_tank::get_volume() {
    return this->core.sum_cell_volumes();
}

vector<double> Fish_tank::volumes(){
    c_loop_all particle_iter(this->core);
    voronoicell cell;
    int particle_id;
    double x, y, z, radius;
    vector<double> volumes;

    if (particle_iter.start()) do if (this->core.compute_cell(cell, particle_iter)) {
        particle_iter.pos(particle_id, x, y, z, radius);  // retrieve the position & id
        volumes.push_back(cell.volume());
    } while (particle_iter.inc());
    return volumes;
}

vector<double> Fish_tank::volumes(py::array_t<int> indices){
    c_loop_all particle_iter(this->core);
    voronoicell cell;
    int particle_id;
    double x, y, z, radius;
    vector<double> volumes;

    auto buf = indices.request();
    int *ptr = (int *) buf.ptr;

    if (particle_iter.start()) do if (this->core.compute_cell(cell, particle_iter)) {
        particle_iter.pos(particle_id, x, y, z, radius);  // retrieve the position & id
        for (int i = 0; i < buf.shape[0]; i++) {
            if (ptr[i] == particle_id){
                volumes.push_back(cell.volume());
            }
        }
    } while (particle_iter.inc());
    return volumes;
}

py::array_t<double> get_voro_volumes(py::array_t<double> points, double z, double c){
    // z: depth of water in the tank
    // c: the fitting parameter, z = c * r^2
    // shape of points in numpy array should be (number, 3)
    size_t inside_count = 0;
    auto buf = points.request();
    size_t index_max = buf.shape[0];
    double *ptr = (double *) buf.ptr;
    Fish_tank tank(z, c);

    double p_x, p_y, p_z;
    for (size_t i = 0; i < index_max; i++){
        p_x = ptr[i * 3 + 0];
        p_y = ptr[i * 3 + 1];
        p_z = ptr[i * 3 + 2];
        if (tank.is_inside(p_x, p_y, p_z)) {
            tank.put(inside_count, p_x, p_y, p_z);
            inside_count++;
        }
    }

    py::array_t<double> volumes_array = py::array_t<double>(inside_count);

    auto buf_vol = volumes_array.request();
    double * ptr_vol = (double *) buf_vol.ptr;
    vector<double> volumes = tank.volumes();

    for (size_t i = 0; i < inside_count; i++) {
        *ptr_vol = volumes[i];
        ptr_vol++;
    }
    return volumes_array;
}

py::array_t<double> get_voro_volumes_select(py::array_t<double> points, py::array_t<int> indices, double z, double c){
    // get voronoi volumes of selected points
    // indices: indices of the points that is selected to calculate cell volumes
    //          all points were used to construct the voronoi tesselation, but only the points of the indices
    //          were considered
    //          Typically, I use this function to get voronoi volumes of cells that do not belong to vertices
    //          of the convex hull
    // z: depth of water in the tank
    // c: the fitting parameter, z = c * r^2
    // shape of points in numpy array should be (number, 3)
    auto buf = points.request();
    size_t index_max = buf.shape[0];
    double *ptr = (double *) buf.ptr;

    Fish_tank tank(z, c);

    double p_x, p_y, p_z;
    for (size_t i = 0; i < index_max; i++){
        p_x = ptr[i * 3 + 0];
        p_y = ptr[i * 3 + 1];
        p_z = ptr[i * 3 + 2];
        if (tank.is_inside(p_x, p_y, p_z)) {
            tank.put(i, p_x, p_y, p_z);
        }
    }

    vector<double> volumes = tank.volumes(indices);

    py::array_t<double> volumes_array = py::array_t<double>(volumes.size());
    auto buf_vol = volumes_array.request();
    double * ptr_vol = (double *) buf_vol.ptr;

    for (size_t i = 0; i < buf_vol.size; i++) {
        ptr_vol[i] = volumes[i];
    }
    return volumes_array;
}


PYBIND11_MODULE(voro, m) {
    m.doc() = "Voronoi analysis using voro++ in 3D";
    m.def("get_voro_volumes", &get_voro_volumes, "getting the voronoi cells volumes in a confined parbobla gemetry");
    m.def("get_voro_volumes_select", &get_voro_volumes_select,
            "getting the voronoi cells volumes in a confined parbobla gemetry, given indices of points inside the convex hull");
}
