#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;


py::array_t<long int> build_tower(py::array_t<long int> &counts){
    auto buf = counts.request();
    long int *ptr = (long int *) buf.ptr;
    long int bin_number = 1;

    for (long int d=0; d < buf.ndim; d++){
        bin_number *= buf.shape[d];
    }

    auto tower = py::array_t<long int> (bin_number+1);
    auto buf_tower = tower.request();
    auto *ptr_tower = (long int *) buf_tower.ptr;

    ptr_tower[0] = 0;

    for (int i=1; i <= bin_number; i++){
        ptr_tower[i] = ptr_tower[i-1] + ptr[i-1];
    }
    return tower;
}


long int get_tower_max(py::array_t<long int> &tower){
    auto buf = tower.request();
    auto *ptr = (long int *) buf.ptr;
    long int size = 1;
    for (auto r : buf.shape){
        size *= r;
    }
    return ptr[size - 1];
}


long int bisection_search(double target, py::array_t<long int> &tower){
    auto buf = tower.request();
    auto *ptr = (long int *) buf.ptr;

    long int size = buf.shape[0];

    long int k_min = 0;
    long int k_max = size;
    long int trial_num = 0;
    long int k = 0;
    
    while (true) {
        k = floor((k_min + k_max) / 2);
        if (ptr[k] < target){
            k_min = k;
        }
        else if (ptr[k-1] > target) {
            k_max = k;
        }
        else {
            return k;
        }
        trial_num += 1;
    }
    return k;
}


py::array_t<long int> tower_sampling(long int size, py::array_t<long int> counts){
    /* 
     * use tower sampling method to sample a ND discrete distribution
     * :param size: the size of the ouput random numbers array
     * :param counts: a ND numpy array, should be the result of counting histogram
     * output: an array of random numbers following the same discrete distribution
     */
    py::array_t<long int> tower = build_tower(counts);
    auto sampling = py::array_t<int> (size);
    auto buf_spl = sampling.request();
    auto *ptr_spl = (int *) buf_spl.ptr;
    long int tower_max = get_tower_max(tower);

    for (long int i=0; i<size; i++){
        double target_val = double(rand()) / RAND_MAX * (tower_max);
        long int target_idx = bisection_search(target_val, tower);
        ptr_spl[i] = target_idx - 1;  // return the left side edge
    }
    return sampling;
}


PYBIND11_MODULE(tower_sample, m){
    m.doc() = "use tower sampling to generate random numbers";
    m.def("tower_sampling", &tower_sampling, "using tower sampling");
    m.def("build_tower", &build_tower, "generate discrete CDF");
    m.def("bisection_search", &bisection_search, "bisection search");
}
