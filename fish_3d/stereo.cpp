#include "stereo.h"

const double RII = 0.750187547;  // inverse of water refractive index
const double RI = 1.333;  // water refractive index

namespace stereo {

Link::Link(int i, int j, int k, double e)
    : indices_{i, j, k}, error_{e} {
        ostringstream r;
        r   << "["  << i << ", " << j  << ", " << k << "], error: "
            << fixed << setprecision(2) << e;
        repr_ = r.str();
}


int& Link::operator[] (int index){
    return indices_[index];
}


Links::Links() : links_{}, size_{0} {}

Links::Links(vector<Link> links)
    :links_{links}, size_{int(links.size())} {
    for (auto link : links_){
        for (int i = 0; i < 3; i++){
            indices_[i].insert(link[i]);
        }
    }
}

Links::Links(PYLinks links_py)
    : links_{}, size_{0} {
        int id_1, id_2, id_3; ///< indices in three views
        double error;
        for (auto link_py : links_py){
            tie(id_1, id_2, id_3, error) = link_py;
            links_.push_back(Link{id_1, id_2, id_3, error});
            size_++;
        }
    }

void Links::report(){
    int idx = 0;
    for (auto sl : links_){
        cout << "#" << ++idx << ": " << sl.repr_ << endl;
    }
    for (int i = 0; i < 3; i++){
        cout << "ID " << i+1 << ": ";
        int count = 0;
        for (auto idx : indices_[i]){
            if (++count < indices_[i].size()) {
                cout << idx << ", ";
            } else {
                cout << idx << endl;
            }
        }
    }
}

Link& Links::operator[] (int index){
    return links_[index];
}


void Links::add(int i, int j, int k, double d){
    bool must_be_new =
        indices_[0].find(i) == indices_[0].end() or
        indices_[1].find(j) == indices_[1].end() or
        indices_[2].find(k) == indices_[2].end();
    if (must_be_new){
        links_.push_back(Link{i, j, k, d});
        indices_[0].insert(i);
        indices_[1].insert(j);
        indices_[2].insert(k);
        size_ ++;
        return;
    } else {
        for (auto& l0 : links_){
            if ((l0[0] == i) and (l0[1] == j) and (l0[2] == k)){
                if (d < l0.error_){
                    l0.error_ = d;
                    ostringstream r;
                    r   << "[" << i << ", " << j << ", " << k << "], error: "
                        << fixed << setprecision(2) << d;
                    l0.repr_ = r.str();
                }
                return;
            }
        }
        links_.push_back(Link{i, j, k, d});
        indices_[0].insert(i);
        indices_[1].insert(j);
        indices_[2].insert(k);
        size_ ++;
        return;
    }
}


void Links::add(Link l){
    bool must_be_new =
        indices_[0].find(l[0]) == indices_[0].end() or
        indices_[1].find(l[1]) == indices_[1].end() or
        indices_[2].find(l[2]) == indices_[2].end();
    if (must_be_new){
        links_.push_back(l);
        indices_[0].insert(l[0]);
        indices_[1].insert(l[1]);
        indices_[2].insert(l[2]);
        return;
    } else {
        for (auto& l0 : links_){
            if ((l0[0] == l[0]) and (l0[1] == l[1]) and (l0[2] == l[2])){
                if (l.error_ < l0.error_){
                    l0.error_ = l.error_;
                    ostringstream r;
                    r   << "[" << l[0] << ", " << l[1] << ", " << l[2] << "], error: "
                        << fixed << setprecision(2) << l.error_;
                    l0.repr_ = r.str();
                }
                return;
            }
        }
        links_.push_back(l);
        indices_[0].insert(l[0]);
        indices_[1].insert(l[1]);
        indices_[2].insert(l[2]);
        return;
    }
}

PYLinks Links::to_py(){
    PYLinks result;
    for (auto link : links_){
        result.push_back(make_tuple(
                    link.indices_[0], link.indices_[1], link.indices_[2], link.error_
                    ));
    }
    return result;
}


Vec3D get_poi(Vec2D& xy, ProjMat& P){
    Vec3D poi;
    double x{xy[0]}, y{xy[1]};
    double p11{P(0, 0)}, p12{P(0, 1)}, p14{P(0, 3)},
           p21{P(1, 0)}, p22{P(1, 1)}, p24{P(1, 3)},
           p31{P(2, 0)}, p32{P(2, 1)}, p34{P(2, 3)};

    poi[0] = (p12*p24 - p12*p34*y - p14*p22 + p14*p32*y + p22*p34*x - p24*p32*x) /
             (p11*p22 - p11*p32*y - p12*p21 + p12*p31*y + p21*p32*x - p22*p31*x);

    poi[1] = -(p11*p24 - p11*p34*y - p14*p21 + p14*p31*y + p21*p34*x - p24*p31*x) /
              (p11*p22 - p11*p32*y - p12*p21 + p12*p31*y + p21*p32*x - p22*p31*x);

    poi[2] = 0;
    return poi;
}

static std::complex<double> complex_sqrt(const std::complex<double> & z) {
    return pow(z, 1. / 2.);
}

static std::complex<double> complex_cbrt(const std::complex<double> & z) {
    return pow(z, 1. / 3.);
}

void solve_quartic(const complex<double> coefficients[5], complex<double> roots[4])
{
    // from https://github.com/sidneycadot/quartic
    // a * x^4 + b * x^3 + c * x^2 + d * x + e == 0

    const std::complex<double> a = coefficients[4];
    const std::complex<double> b = coefficients[3] / a;
    const std::complex<double> c = coefficients[2] / a;
    const std::complex<double> d = coefficients[1] / a;
    const std::complex<double> e = coefficients[0] / a;

    const std::complex<double> Q1 = c * c - 3. * b * d + 12. * e;
    const std::complex<double> Q2 = 2. * c * c * c - 9. * b * c * d + 27. * d * d + 27. * b * b * e - 72. * c * e;
    const std::complex<double> Q3 = 8. * b * c - 16. * d - 2. * b * b * b;
    const std::complex<double> Q4 = 3. * b * b - 8. * c;

    const std::complex<double> Q5 = complex_cbrt(Q2 / 2. + complex_sqrt(Q2 * Q2 / 4. - Q1 * Q1 * Q1));
    const std::complex<double> Q6 = (Q1 / Q5 + Q5) / 3.;
    const std::complex<double> Q7 = 2. * complex_sqrt(Q4 / 12. + Q6);

    roots[0] = (-b - Q7 - complex_sqrt(4. * Q4 / 6. - 4. * Q6 - Q3 / Q7)) / 4.;
    roots[1] = (-b - Q7 + complex_sqrt(4. * Q4 / 6. - 4. * Q6 - Q3 / Q7)) / 4.;
    roots[2] = (-b + Q7 - complex_sqrt(4. * Q4 / 6. - 4. * Q6 + Q3 / Q7)) / 4.;
    roots[3] = (-b + Q7 + complex_sqrt(4. * Q4 / 6. - 4. * Q6 + Q3 / Q7)) / 4.;
}


double get_u_simple(double d, double x, double z){
    x = x / 1000; d = d / 1000; z = z / 1000; // mm -> m
    double d2 = d * d;
    double x2 = x * x;
    double z2 = z * z;
    double n2 = RI * RI;

    double A = n2   - 1;  // u^4
    double B = - 2 * A * x; // u^3
    double C = d2   * n2   + A * x2   - z2;  // u^2
    double D = -2 * d2   * n2   * x;  // u^1
    double E = d2   * n2   * x2;

    double p1 = 2 * pow(C, 3) - 9*B*C*D + 27*A*D*D  + 27*B*B*E  - 72*A*C*E;
    double p2 = p1 +    sqrt(-4 * pow(C*C  - 3*B*D + 12*A*E, 3) + p1*p1);
    double p3 = (C*C  - 3*B*D + 12*A*E) / (3*A * pow(p2/2, 1.0/3.0)) + pow(p2/2, 1.0/3.0) / (3*A);
    double p4 =    sqrt(B*B  / (4*A*A)  - (2*C)/(3*A) + p3);
    double p5 = B*B / (2*A*A)   - (4*C)/(3*A) - p3;
    double p6 = (-pow(B/A, 3) + (4*B*C)/(A*A) - 8*D/A) / (4*p4);

    double u = -100;

    if ((p5 - p6) > 0){
        u = -B / (4 * A) - p4 / 2 - sqrt(p5 - p6)/2;
        if ((u > 0) and (u <= x)){
            return u * 1000;
        }
        u = -B / (4 * A) - p4 / 2 + sqrt(p5 - p6) / 2;
        if ((u > 0) and (u <= x)){
            return u * 1000;
        }
    }
    if ((p5 + p6) > 0){
        u = -B / (4 * A) + p4 / 2 - sqrt(p5 + p6) / 2;
        if ((u > 0) and (u <= x)){
            return u * 1000;
        }

        u = -B / (4 * A) + p4 / 2 + sqrt(p5 + p6) / 2;
        if ((u > 0) and (u <= x)){
            return u * 1000;
        }
    }
    return u;
}


double get_u(double d, double x, double z){
    double n2 = RI * RI;
    complex<double> coef[5];
    complex<double> roots[4];
    coef[4] = {n2 - 1, 0};
    coef[3] = {2 * x - 2 * n2 * x, 0};
    coef[2] = {d*d * n2 - x*x + n2 * x*x - z*z, 0};
    coef[1] = {-2 * d*d * n2 * x, 0};
    coef[0] = {d*d * n2 * x*x, 0};
    solve_quartic(coef, roots);
    for (auto r : roots){
        if ((r.imag() == 0) and (r.real() > 0) and (r.real() < x)){
            return r.real();
        }
    }
    throw("root finding failed");
}

Vec3D get_refraction_ray(Vec3D incidence){
    Vec3D refraction;
    incidence /= incidence.norm();
    double cos_incid = -incidence[2];  // cos = incidence dot (0, 0, 1)
    double sin_ref_2 = RII * RII  * (1 - cos_incid * cos_incid);
    // t = rri * i + (rri * cos_i - np.sqrt(1 - sin_t_2)) * n
    refraction = RII * incidence;
    refraction[2] += RII * cos_incid - sqrt(1 - sin_ref_2);
    refraction /= refraction.norm();
    return refraction;
}


Vec3D get_intersection(Lines lines){
    Vec3D intersection, c1, c2, b;  // shape (3, 1)
    Mat33 tmp, M;
    M.setZero(); b.setZero();
    for (auto line : lines){
        c1 = line.row(0);
        c2 = line.row(1);
        tmp = c2 * c2.transpose() - c2.dot(c2) * Mat33::Identity();
        M += tmp;
        b += tmp * c1;
    }
    return M.ldlt().solve(b);
}


Vec2D reproject_refractive(Vec3D point, ProjMat P, Vec3D camera_origin){
    Vec2D oq, xy;
    Vec3D xyh;
    Vec3DH poih;
    double d = abs(camera_origin[2]);
    double z = abs(point[2]);
    double x = (camera_origin - point).topRows(2).norm();
    double u = get_u_simple(d, x, z);
    oq = (point - camera_origin).topRows(2);
    oq /= oq.norm();
    poih.topRows(2) = camera_origin.topRows(2) + u * oq;
    poih[2] = 0;
    poih[3] = 1;
    xyh = P * poih;
    xyh /= xyh[2];
    xy = xyh.topRows(2);
    return xy;
}


double get_reproj_error(Vec3D xyz, TriXY centres, TriPM Ps, TriXYZ Os){
    double error{0}; 
    Vec2D reproj;
    for (int i = 0; i < 3; i++){
        reproj = reproject_refractive(xyz, Ps[i], Os[i]);
        error += (centres[i] - reproj).norm();
    }
    return error / 3;
}


double get_error(TriXY& centres, TriPM& Ps, TriXYZ& Os){
    double error{0}; 
    Lines lines;
    Line l;
    Vec3D poi, xyz;
    Vec2D reproj;
    for (int i = 0; i < 3; i++){
        poi = get_poi(centres[i], Ps[i]);
        l.row(0) = poi;
        l.row(1) = get_refraction_ray(poi - Os[i]);
        lines[i] = l;
    }
    xyz = get_intersection(lines);
    for (int i = 0; i < 3; i++){
        reproj = reproject_refractive(xyz, Ps[i], Os[i]);
        error += (centres[i] - reproj).norm();
    }
    return error / 3;
}


double get_error_with_xyz(TriXY& centres, TriPM& Ps, TriXYZ& Os, Vec3D& xyz){
    double error{0};
    Vec2D reproj;
    for (int i = 0; i < 3; i++){
        reproj = reproject_refractive(xyz, Ps[i], Os[i]);
        error += (centres[i] - reproj).norm();
    }
    return error / 3;
}


Vec3D three_view_reconstruct(TriXY Cs, TriPM Ps, TriXYZ Os){
    Vec3D poi, refraction, xyz;
    Lines lines;
    Line line;
    for (int view = 0; view < 3; view++){
        poi = get_poi(Cs[view], Ps[view]);
        refraction = get_refraction_ray(poi - Os[view]);
        line.row(0) = poi;
        line.row(1) = refraction;
        lines[view] = line;
    }
    return get_intersection(lines);
}


Links three_view_match(
        Coord2D& centres_1, Coord2D& centres_2, Coord2D& centres_3,
        ProjMat P1, ProjMat P2, ProjMat P3,
          Vec3D O1,   Vec3D O2,   Vec3D O3,
        double tol_2d, bool optimise=true
        ){
    Vec2D x1, x2, x3;
    Vec3D poi_1, poi_2, poi_3, ref_1, ref_2, ref_3, xyz;
    Line l1, l2, l3;
    int n1 = centres_1.rows();
    int n2 = centres_2.rows();
    int n3 = centres_3.rows();
    double error{0};

    Links links;

    for (int i = 0; i < n1; i++){
        x1 = centres_1.row(i);
        poi_1 = get_poi(x1, P1);
        ref_1 = get_refraction_ray(poi_1 - O1);
        l1.row(0) = poi_1;
        l1.row(1) = ref_1;
        for (int j = 0; j < n2; j++){
            x2 = centres_2.row(j);
            poi_2 = get_poi(x2, P2);
            ref_2 = get_refraction_ray(poi_2 - O2);
            l2.row(0) = poi_2;
            l2.row(1) = ref_2;
            for (int k = 0; k < n3; k++){
                x3 = centres_3.row(k);
                poi_3 = get_poi(x3, P3);
                ref_3 = get_refraction_ray(poi_3 - O3);
                l3.row(0) = poi_3;
                l3.row(1) = ref_3;
                xyz = get_intersection(Lines{l1, l2, l3});
                error = get_reproj_error(
                            xyz,
                            TriXY{x1, x2, x3},
                            TriPM{P1, P2, P3},
                            TriXYZ{O1, O2, O3}
                        );
                if (error < tol_2d){
                    links.add(i, j, k, error);
                }
            }
        }
    }
    if (optimise){
        return optimise_links(links);
    } else {
        return links;
    }
}


IloBoolVarArray get_variables(IloEnv& env, Links system){
    IloBoolVarArray x(env);
    for (int i = 0; i < system.size_; i++){
        x.add(IloBoolVar(env));
    }
    return x;
}


IloRangeArray get_constrains(IloEnv& env, IloBoolVarArray& x, Links sys){
    IloRangeArray constrains(env);
    IloInt idx;
    for (int view = 0; view < 3; view++) { ///< ∀ i
        for (auto i : sys.indices_[view]) {
            IloExpr sum(env);
            idx = 0;
            for (auto link : sys.links_){
                if (link[view] == i){
                    sum += x[idx];  ///< ∑(jk)[ x(ijk) ] ≥ 1
                }
                idx++;
            }
            constrains.add(sum >= 1);
        }
    }
    return constrains;
}


IloExpr get_cost(IloEnv& env, IloBoolVarArray x, Links system){
    IloExpr cost(env);
    for (int i = 0; i < system.size_; i++){
        cost += system[i].error_ * x[IloInt(i)];
    }
    return cost;
}


Links optimise_links(Links system){
    Links new_system;
    IloEnv   env;
    IloModel model(env);

    IloBoolVarArray x = get_variables(env, system);  // variables
    IloRangeArray constrains = get_constrains(env, x, system);
    IloExpr cost = get_cost(env, x, system);
    model.add(IloMinimize(env, cost));
    model.add(constrains);

    IloCplex cplex(model);
    cplex.setOut(env.getNullStream());  ///< avoid log msgs on the screen
    if ( !cplex.solve() ) {
     env.error() << "Failed to optimize LP" << endl;
     throw(-1);
    }

    IloNumArray vals(env);
    cplex.getValues(vals, x);
    for (int i = 0; i < system.size_; i++){
        if (vals[i] == 1){
            new_system.add(system[i]);
        }
    }
    env.end();
    return new_system;
}

}
