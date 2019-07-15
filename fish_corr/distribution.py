import numpy as np
from scipy.optimize import root_scalar


class Tank
    def __init__(self, base, surface):
        """
        A fish tank in 3D, whose surface can be described by
            Z = c1 * (exp(c2 * r) - 1)
        where c1 = 4.0434 * 10^-3, c2 = 2.3497 * 10^2
        """
        self.align_mat = self.__get_align_mat(base, surface)


    def z(self, r):
        return 2.4732e2 * (np.exp(4.0434e-3 * r) - 1)

    def __get_align_mat(self, ):
        return points


    def get_distance(self, points):
        XYZ = points.T
        X, Y, Z = XYZ
        theta = np.arctan(Y / X)
        radii = np.sqrt(X**2 + Y**2)
        proj_cylind = np.empty((3, len(X)), dtype=np.float64)
        for i, (r, z) in enumerate(zip(radii, Z)):
            par_func = lambda x:\
                2(x - r) + 0.200003 * np.exp(0.0040434 * x) *\
                (24.732 * (np.exp(0.0040434 * x) - 1) - z)
            r_project = root_scalar(par_func, x0=r)
            z_project = self.z(r_project)
            proj_cylind[0, i] = r_project
            proj_cylind[1, i] = theta[i]
            proj_cylind[2, i] = z_project
        proj_cart = np.empty((3, len(X)), dtype=np.float64)
        proj_cart[0] = proj_cylind[0] * np.cos(proj_cylind[1])  # x
        proj_cart[1] = proj_cylind[0] * np.sin(proj_cylind[1])  # y
        proj_cart[2] = proj_cylind[2]  # z
        dists = np.linalg.norm(proj_cart - XYZ, axis=0)
        return dists
