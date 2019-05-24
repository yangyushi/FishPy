#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <map>

namespace py = pybind11;

using std::cout;
using std::map;
using std::vector;
using std::string;

float get_intersect_of_lines(vector<map<string, vector<float>>> lines){
    int line_num = lines.size();
    for (auto line : lines){
        cout << line.find("unit")->second << '\n';
    }
}

PYBIND11_MODULE(ray_trace, m){
    m.doc() = "refractive ray tracing";
    m.def("get_intersect_of_lines", &get_intersect_of_lines,
          "calculate the point that are closest to multipl lines");
}
