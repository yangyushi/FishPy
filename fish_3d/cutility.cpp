#include <iostream>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <set>
using namespace std;
namespace py = pybind11;

using Pair = vector<int>;
using Pairs = vector<Pair>;

bool should_join(Pair p1, Pair p2){
    for (int n1 : p1){
    for (int n2 : p2){
        if (n1 == n2) {return true;}
    }}
    return false;
}

Pair join_pair(Pair p1, Pair p2){
    set<int> unique;
    Pair joined;
    for (int num : p1) {unique.insert(num);}
    for (int num : p2) {unique.insert(num);}
    copy(unique.begin(), unique.end(), back_inserter(joined));
    return joined;
}


/**
*   [(2, 3), (3, 5), (2, 6), (8, 9), (9, 10)]
*    --->
*   [(2, 3, 5, 6), (8, 9, 10)]
*/
void join_pairs_inplace(Pairs& pairs){
    int length = pairs.size();
    Pair p1, p2;
    for (int i1 = 0; i1 < length; i1++){
        for (int i2 = 0; i2 < length; i2++){
            p1 = pairs[i1];
            p2 = pairs[i2];
            if ((i1 != i2) and (should_join(p1, p2))){
                pairs.push_back(join_pair(p1, p2));
                pairs.erase(pairs.begin() + i1);
                pairs.erase(pairs.begin() + i2 - 1);
                join_pairs_inplace(pairs);
                return;
            }
        }
    }
}

Pairs join_pairs(Pairs& pairs){
    Pairs joined;
    copy(pairs.begin(), pairs.end(), back_inserter(joined));
    join_pairs_inplace(joined);
    return joined;
}


PYBIND11_MODULE(cutility, m){
    m.doc() = "helper functions optimised in cpp";
    m.def("join_pairs", &join_pairs, py::arg("pairs"));
    }
