#include "stereo.h"

const double RII = 0.751879699;  // inverse of water refractive index
const double RI = 1.33;  // water refractive index


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


double get_u(double d, double x, double z){
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
    double u = get_u(d, x, z);
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


double get_error(TriXY centres, TriPM Ps, TriXYZ Os){
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


Links three_view_match(
        Coord2D& centres_1, Coord2D& centres_2, Coord2D& centres_3,
        ProjMat P1, ProjMat P2, ProjMat P3,
        Vec3D O1, Vec3D O2, Vec3D O3, double tol_2d){
    Links result;
    Vec2D x1, x2, x3;
    Vec3D poi_1, poi_2, poi_3, ref_1, ref_2, ref_3, xyz;
    Line l1, l2, l3;
    int n1 = centres_1.rows();
    int n2 = centres_2.rows();
    int n3 = centres_3.rows();
    double error{0};

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
                    result.push_back(Link{i, j, k});
                }
            }
        }
    }
    return result;
}
