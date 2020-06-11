#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

using std::vector;
namespace py = pybind11; 

static constexpr int LinkMatSize = Eigen::Dynamic;
using LinkMat = Eigen::Matrix<bool, LinkMatSize, LinkMatSize, Eigen::RowMajor>;
using Config = vector<int>;

vector<int> nonzero(LinkMat& lm, int row_num) {
    // return the non zero indices in a given row
    vector<int> indices;
    int n_col = lm.cols();
    for (int i = 0; i < n_col; i++){
        if (lm(row_num, i)) {
            indices.push_back(i);
        }
    }
    return indices;
}


int count_config(Config& config){
    /*
    * count the non-zero elements in a configuration
    * the result is possible links in a configuration
    */
    int result = 0;
    for (auto col : config){
        if (col >= 0){
            result += 1;
        }
    }
    return result;
}

bool conflict(Config& config){
    if (config.size() == 1) {
        return false;
    }
    else{
        int last_index = config.size() - 1;
        for (int i = 0; i < last_index; i++) {
            if (config[i] == config[last_index]) { return true; }
        }
    }
    return false;
}


bool conflict_future_row(LinkMat& lm, int row, int col){
    for (int r = row+1; r < lm.rows(); r++){
        if (lm(r, col) > 0){
            return true;
        }
    }
    return false;
}


void solve(LinkMat& lm, vector<Config>& solutions,
           int mat_rank, int row_num=0, int level=0, Config config={}){
    if (level == mat_rank) {
        solutions.push_back(config);
        return;
    } else {
        if (lm.rows() <= row_num) { return; }
        while (lm.row(row_num).count() == 0) {  // jump through blank rows
            config.push_back(-1);
            row_num++;
            if (lm.rows() <= row_num) { return; }
        }
        config.push_back(-1);
        vector<int> alternative = config;  // choose nothing in this row
        for (int col : nonzero(lm, row_num)) {
            config.back() = col;
            if ( not conflict(config) ) {
                if (conflict_future_row(lm, row_num, col)) {
                    solve(lm, solutions, mat_rank, row_num+1, level, alternative);
                }
                solve(lm, solutions, mat_rank, row_num+1, level+1, config);
            }
        }
    }
    return;
}


void solve_dense(LinkMat& lm, vector<Config>& solutions, const int& max_row,
        int row_num=0, Config config={}){
    /*
     * lm: the linkage matrix, the binarized distance matrix
     * solutions: all possible link configurations
     */
    if (row_num == max_row) {
        solutions.push_back(config);
        return;
    }
    else {
        vector<int> possible_cols;
        config.push_back(-1);
        Config alternative{config};

        for (int col : nonzero(lm, row_num)) { 
            config.back() = col;
            if ( not conflict(config) ) { possible_cols.push_back(col); }
        }

        for (int col : possible_cols) {  // choose nothing in this row
            if (conflict_future_row(lm, row_num, col)) {
                solve_dense(lm, solutions, max_row, row_num+1, alternative);
                break;
            }
        }

        for (int col : possible_cols) { // select different columns in this row
            config.back() = col;
            solve_dense(lm, solutions, max_row, row_num+1, config);
        }
    }
}


py::array_t<int> solve_nrook(LinkMat lm){
    Eigen::FullPivLU<Eigen::MatrixXf> dec(lm.cast<float>());
    int lm_rank = dec.rank();
    vector<vector<int>> solutions;
    int row_idx = 0;

    solve(lm, solutions, lm_rank);

    int num_solutions = solutions.size();

    auto result = py::array_t<int>(num_solutions * lm_rank * 2);
    auto buffer = result.request();
    int *ptr = (int *) buffer.ptr;

    for (auto config : solutions) {
        row_idx = 0;
        for (int col : config) {
            if (col >= 0){
                *ptr++ = row_idx;
                *ptr++ = col;
            }
        row_idx++;
        }
    }
    result.resize({num_solutions, lm_rank, 2});
    return result;
}


py::array_t<int> solve_nrook_dense(LinkMat lm, int max_row){
    vector<vector<int>> solutions;
    int row_idx = 0;
    int size = 0;

    solve_dense(lm, solutions, max_row);

    int max_conf_size = 0;
    for (auto config : solutions) {
        size = count_config(config);
        if (size > max_conf_size){
            max_conf_size = size;
        }
    }

    int num_solutions = 0;
    for (auto config : solutions) {
        if (count_config(config) == max_conf_size){
            num_solutions += 1;
        }
    }

    auto result = py::array_t<int>(num_solutions * max_conf_size * 2);
    auto buffer = result.request();
    int *ptr = (int *) buffer.ptr;

    for (auto config : solutions) {
        if (count_config(config) == max_conf_size) {
            row_idx = 0;
            for (int col : config) {
                if (col >= 0){
                    *ptr++ = row_idx;
                    *ptr++ = col;
                }
            row_idx++;
            }
        }
    }
    result.resize({num_solutions, max_conf_size, 2});
    return result;
}


PYBIND11_MODULE(nrook, m){
    m.doc() = "solve n rook problem in restricted available sites";

    m.def("solve_nrook", &solve_nrook, "solve the n rook problem in a given availabel sites",
          py::return_value_policy::move, py::arg("lm").noconvert());

    m.def("solve_nrook_dense", &solve_nrook_dense,
            "solve the n rook problem in a given availabel sites from a dense distance matrix",
          py::return_value_policy::move, py::arg("lm").noconvert(), py::arg("max_row"));
}
