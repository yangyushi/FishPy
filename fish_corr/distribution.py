import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


class Tank:
    def __init__(self, base, surface):
        """
        A fish tank in 3D, whose surface can be described by
            Z = c1 * (exp(c2 * r) - 1)
        where c1 = 2.3497 * 10^1, c2 = 4.0434 * 10^-3
        """
        self.base = np.vstack(base)  # shape: (3, 1)
        self.rot = self.__get_rotation(base, surface)

    def z(self, r):
        return 2.3497e1 * (np.exp(4.0434e-3 * r) - 1)

    def __get_rotation(self, base, surface):
        tank = np.vstack((surface, base)).T
        covar = np.cov(tank)
        u, s, vh = np.linalg.svd(covar)
        return vh

    def get_projection(self, points):
        XYZ = self.rot @ (points.T - self.base)
        X, Y, Z = XYZ
        theta = np.arctan(Y / X)
        theta[(X<0) & (Y>0)] += np.pi
        theta[(X<0) & (Y<0)] -= np.pi
        radii = np.sqrt(X**2 + Y**2)
        proj_cylind = np.empty((3, len(X)), dtype=np.float64)

        for i, (r, z) in enumerate(zip(radii, Z)):
            par_func = lambda x: 4.4648 * np.exp(0.0080868 * x) + 2 * (x - r) + np.exp(0.004034 * x) * (-4.4648 - 0.190016 * z)
            r_project = root_scalar(par_func, x0=r, x1=r/2).root
            z_project = self.z(r_project)

            proj_cylind[0, i] = r_project
            proj_cylind[1, i] = theta[i]
            proj_cylind[2, i] = z_project

        proj_cart = np.empty((3, len(X)), dtype=np.float64)
        proj_cart[0] = proj_cylind[0] * np.cos(theta)  # x = r * cos(t)
        proj_cart[1] = proj_cylind[0] * np.sin(theta)  # y = r * sin(t)
        proj_cart[2] = proj_cylind[2]  # z = z
        return (self.rot.T @ proj_cart) + self.base  # (3, n)

    def get_cylinder(self, points):
        XYZ = self.rot @ (points.T - self.base)
        X, Y, Z = XYZ
        theta = np.arctan(Y / X)
        theta[(X<0) & (Y>0)] += np.pi
        theta[(X<0) & (Y<0)] -= np.pi
        radii = np.sqrt(X**2 + Y**2)
        return np.array([radii, theta, Z])
