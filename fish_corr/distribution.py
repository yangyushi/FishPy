import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def a2r_cost(x, a, b, y):
    r_range = np.arange(x)
    curve = np.sqrt(a**2 * b**2 * np.exp(2 * b * r_range) + 1)
    length = np.trapz(curve, r_range)
    return length - y


def reduce_angle_gap(angles):
    """
    angles should be in between 0 to 2pi
    """
    sa = np.sort(angles)  # sorted angles
    gaps = sa[1:] - sa[:-1]
    left_gap = sa[0]
    right_gap = 2 * np.pi - sa[-1]
    if np.max(gaps) > (left_gap + right_gap):
        oi = np.argmax(gaps)
        origin = (sa[oi] + sa[oi + 1]) / 2
    else:
        origin = 0
    shifted = angles - origin
    shifted[shifted < 0] += np.pi * 2
    return shifted, origin


class Tank:
    def __init__(self, base, surface):
        """
        A fish tank in 3D, whose surface can be described by
            Z = c1 * (exp(c2 * r) - 1)
        where c1 = 2.3497 * 10^1, c2 = 4.0434 * 10^-3
        """
        self.base = np.vstack(base)  # shape: (3, 1)
        self.rot = self.__get_rotation(base, surface)
        self.c1 = 23.497
        self.c2 = 4.0434e-3
        self.z_max = -base[-1]  # maximum height from bottom of the tank
        self.r_max = np.log(self.z_max * 1/self.c1 + 1) / self.c2

    def z(self, r):
        return self.c1 * (np.exp(self.c2 * r) - 1)

    def __get_rotation(self, base, surface):
        tank = np.vstack((surface, base)).T
        covar = np.cov(tank)
        u, s, vh = np.linalg.svd(covar)
        return vh

    def get_cylinder(self, points):
        XYZ = self.rot @ (points.T - self.base)
        X, Y, Z = XYZ
        theta = np.arctan(Y / X)
        theta[(X<0) & (Y>0)] += np.pi
        theta[(X<0) & (Y<0)] -= np.pi
        theta[theta < 0] += np.pi * 2
        radii = np.sqrt(X**2 + Y**2)
        return np.array([radii, theta, Z])

    def get_projection(self, points):
        radii, theta, Z = self.get_cylinder(points)
        proj_cylind = np.empty((3, len(Z)), dtype=np.float64)

        for i, (r, z) in enumerate(zip(radii, Z)):
            par_func = lambda x: 4.4648 * np.exp(0.0080868 * x) + 2 * (x - r) + np.exp(0.004034 * x) * (-4.4648 - 0.190016 * z)
            r_project = root_scalar(par_func, x0=r, x1=r/2).root
            z_project = self.z(r_project)

            proj_cylind[0, i] = r_project
            proj_cylind[1, i] = theta[i]
            proj_cylind[2, i] = z_project

        proj_cart = np.empty((3, len(Z)), dtype=np.float64)
        proj_cart[0] = proj_cylind[0] * np.cos(theta)  # x = r * cos(t)
        proj_cart[1] = proj_cylind[0] * np.sin(theta)  # y = r * sin(t)
        proj_cart[2] = proj_cylind[2]  # z = z
        return (self.rot.T @ proj_cart) + self.base  # (3, n)

    def get_curvilinear(self, points):
        radii, theta, Z = self.get_cylinder(points)
        curvilinear = np.empty((3, len(Z)), dtype=np.float64)
        a, b = self.c1, self.c2

        for i, (r, z) in enumerate(zip(radii, Z)):
            par_func = lambda x: 4.4648 * np.exp(0.0080868 * x) + 2 * (x - r) + np.exp(0.004034 * x) * (-4.4648 - 0.190016 * z)
            r_project = root_scalar(par_func, x0=r, x1=r/2).root
            z_project = self.z(r_project)
            distance = np.sqrt((r_project - r)**2 + (z_project - z)**2)

            r_range = np.arange(0, abs(r_project))
            curve = np.sqrt(a**2 * b**2 * np.exp(2 * b * r_range) + 1)
            arc_length = np.trapz(curve, r_range)
            curvilinear[0, i] = arc_length
            curvilinear[1, i] = theta[i]
            curvilinear[2, i] = distance
        return curvilinear

    def random(self, number):
        """
        random points inside tank
        use rejection method to generate random points inside the tank
        """
        a, b = self.c1, self.c2
        succeed = False
        while not succeed:
            x, y, z = np.random.random((3, number * 10)) * np.vstack((self.r_max, self.r_max, self.z_max))
            mask = np.zeros(len(z), dtype=bool)
            r = np.sqrt(x**2 + y**2)
            mask[z > self.z(r)]= True
            if 10 * number - np.sum(mask) > number:
                x_inside = x[mask]
                y_inside = y[mask]
                r_inside = np.sqrt(x_inside**2 + y_inside**2)
                theta = np.random.uniform(-np.pi, np.pi, len(x_inside))
                x_inside = r_inside * np.cos(theta)
                y_inside = r_inside * np.sin(theta)
                z_inside = z[mask]
                random_points = np.vstack((x_inside, y_inside, z_inside)).T
                succeed = True
        return random_points[:number]

    def random_curvilinear(self, points):
        """
        generate random points in a box in the curvilinear coordinate (L, θ, D)
        L - curvilinear length
        θ - azimuth angle
        D - cloest distance from fish to tank
        """
        num = len(points)
        L, T, D = self.get_curvilinear(points)
        dT, origin = reduce_angle_gap(T)
        L_rand = np.sqrt(np.random.random(num)) * (L.max() - L.min()) + L.min()
        dT_rand = np.random.uniform(dT.min(), dT.max(), num)
        D_rand = np.random.uniform(D.min(), D.max(), num)
        R_rand, Z_rand = self.curvilinear_2_cylindar(L_rand, D_rand)
        T_rand = (dT + origin) % (np.pi * 2)
        return np.vstack([
            R_rand * np.cos(T_rand),
            R_rand * np.sin(T_rand),
            Z_rand,
        ]).T

    def curvilinear_2_cylindar(self, L, D):
        R = np.empty(len(L))
        Z = np.empty(len(L))
        a, b = self.c1, self.c2
        for i, l in enumerate(L):
            rp = root_scalar(a2r_cost, args=(a, b, l), x0=l, x1=l/2).root
            zp = self.z(rp)
            s = a*b*np.exp(b*rp)  # slope
            tanget = (1, s)
            normal = (1, -1/s)
            theta = np.arccos(-s / np.sqrt(2 + s**2))
            dr = D[i] * np.cos(theta)
            dz = D[i] * np.sin(theta)
            R[i] = rp + dr
            Z[i] = zp + dz
        return R, Z
