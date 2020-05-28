#include "stereo.h"
#include "temporal.h"
#include <iostream>


temporal::PYTrajs get_trajectories(
        vector<temporal::Coord2D> frames, double search_range
        ){
    temporal::PYTrajs result;
    temporal::Trajs trajs = temporal::link_2d(frames, search_range, false);
    for (auto t : trajs){
        result.push_back( temporal::PYTraj{t.indices_, t.time_} );
    }
    return result;
}


int main(){
    //cout << "Before optimisation: " << endl;
    temporal::Links links;
    links.add(0, 0, 2); links.add(0, 1, 1); links.add(1, 0, 1);
    links.add(1, 1, 1); links.add(0, 0, 0); links.report();
    /*
     * optimised links should be [0, 0], [1, 1], [2, 2]
     */
    cout << "After optimisation: " << endl;
    temporal::Links new_links = optimise_links(links);
    new_links.report();

    temporal::Coord2D f0{3, 2};
    f0 << 0.0, 0.0, 1.0, 1.0, 4.0, 4.0;
    temporal::Coord2D f1{3, 2};
    f1 << 0.2, 0.2, 1.2, 1.2, 4.2, 4.2;
    temporal::Coord2D f2{3, 2};
    f2 << 0.4, 0.4, 1.4, 1.4, 4.4, 4.4;
    temporal::Coord2D f3{3, 2};
    f3 << 0.5, 0.5, 1.5, 1.5, 4.5, 4.5;
    vector<temporal::Coord2D> frames {f0, f1, f2, f3};

    temporal::PYTrajs trajs = get_trajectories(frames, 2.4);
    cout << trajs.size() << " trajectories found" << endl;

    for (auto t : trajs){
        cout << "traj: ";
        for (auto num : t[0]){
            cout << num << ", ";
        }
        cout << endl;
    }

}
