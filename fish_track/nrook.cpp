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

vector<int> nonzero(LinkMat & lm, int row_num) {
    // return the non zero indices in a given row
    auto row = lm.row(row_num);
    vector<int> indices;
    int n_col = lm.cols();
    for (int i = 0; i < n_col; i++){
        if (row[i]) { indices.push_back(i); }
    }
    return indices;
}


bool conflict(Config config){
    int last_index = config.size() - 1;
    for (int i = 0; i < config.size() - 1; i++) {
        if (config[i] == config[last_index]) { return true; }
    }
    return false;
}


bool conflict_future_row(LinkMat & lm, int row, int col){
    for (int r = row+1; r < lm.rows()-1; r++){
        if (lm(r, col) > 0){
            return true;
        }
    }
    return false;
}


void solve(LinkMat & lm, vector<Config> & solutions,
           int mat_rank, int row_num=0, int level=0, Config config={}){
    if (level == mat_rank) {
        solutions.push_back(config);
        return;
    } else {
        if (lm.rows() <= row_num) { return; }
        while (lm.row(row_num).count() == 0) {
            config.push_back(-1);
            row_num++;
            if (lm.rows() <= row_num) { return; }
        }
        config.push_back(-1);
        vector<int> alternative = config;
        for (int col : nonzero(lm, row_num)) {
            config.back() = col;
            if ( not conflict(config) ) {
                if (conflict_future_row(lm, row_num, col)) {
                    solve(lm, solutions, mat_rank, row_num+1, level  , alternative);
                }
                solve(lm, solutions, mat_rank, row_num+1, level+1, config);
            }
        }
    }
    return;
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

PYBIND11_MODULE(nrook, m){
    m.doc() = "solve n rook problem in restricted available sites";

    m.def("solve_nrook", &solve_nrook, "solve the n rook problem in a given availabel sites",
          py::return_value_policy::move, py::arg("lm").noconvert());
}
