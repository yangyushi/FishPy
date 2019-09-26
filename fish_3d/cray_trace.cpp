#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <array>

namespace py = pybind11;
using namespace Eigen;
using std::array;

py::array_t<double> get_intersect_of_lines(py::array_t<double> py_lines){
    /*
     * The shape of py_lines is (N, view_number, 2, 3)
     * py_lines would results in N 3d locations (xyz)
     * for each 3d location, there is one line in each view
     * the line is represented as a point and a unit vector direction in 3D, in the shape of (2, 3)
     */
    auto buffer = py_lines.request();
    int num = buffer.shape[0];
    int view = buffer.shape[1];
    double *ptr = (double *) buffer.ptr;
    array<double, 6> arr;
    Matrix3d M; 
    Vector3d b; 
    Matrix3d tmp;
    Vector3d a;
    Vector3d xyz;
    auto result = py::array_t<double>(num * 3);
    auto buffer_result = result.request();  // result buffer
    auto *ptr_result = (double *) buffer_result.ptr;

    for (int ip = 0; ip < num; ip++){        // index of 3D point
        M << 0, 0, 0, 0, 0, 0, 0, 0, 0;
        b << 0, 0, 0;
        for (int iv = 0; iv < view; iv++) {  // index of view
            for (int i=0; i < 6; i++){ // first 3 is a 3D point on the line, last 3 is line direction
                arr[i] = *ptr;
                ptr++;
            }
            tmp << -arr[4]*arr[4] - arr[5]*arr[5], arr[3] * arr[4], arr[3] * arr[5],
                   arr[4] * arr[3], -arr[3]*arr[3] - arr[5]*arr[5], arr[4] * arr[5],
                   arr[5] * arr[3], arr[5] * arr[4], -arr[3]*arr[3] - arr[4]*arr[4];
            a << arr[0], arr[1], arr[2];
            M = M + tmp;
            b = b + tmp * a;
        }
        xyz = M.ldlt().solve(b);
        for (int i=0; i<3; i++){
            *ptr_result = xyz[i];
            ptr_result++;
        }
    }
    result.resize({num, 3});
    return result;
}

PYBIND11_MODULE(cray_trace, m){
    m.doc() = "refractive ray tracing";
    m.def("get_intersect_of_lines", &get_intersect_of_lines,
          "calculate the point that are closest to multipl lines",
          py::return_value_policy::move, py::arg("lines").noconvert());
}
