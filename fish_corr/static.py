import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import root_scalar
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import binned_statistic_2d
from numba import njit, prange
try:
    from fish_3d.utility import box_count_polar_image, box_count_polar_video
except ImportError:
    pass

def a2r_cost(x, c, y):
    length = 0.5 * x * np.sqrt(1 + 4 * c**2 * x**2) +\
            np.arcsinh(2 * c * x) / (4 * c)
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
    def __init__(self, base, surface=None):
        """
        A fish tank in 3D, whose surface can be described by
            z = c * r ^ 2
        where c = 0.7340428, and the length unit for the equation is m [meter]
        """
        self.base = np.vstack(base)  # shape: (3, 1)
        if isinstance(surface, type(None)):
            self.rot = np.identity(3)
        else:
            self.rot = self.__get_rotation(base, surface)
        self.c = 0.7340428
        self.z_max = -base[-1]  # maximum height from bottom of the tank
        self.r_max = np.sqrt(self.z_max / self.c * 1000)

    def __project_r(self, radii, z):
        """
        calculate the projection of 2D point in cylindar coordinate system
        to a hyprobolic function y = self.c * x^2
        the input / ouput unit is mm
        in the calculation, the unit is convert to m
        """
        r = radii / 1000  # mm -> m
        z = z.copy() / 1000
        p = 2**(1/3)
        term = 108 * self.c ** 4 * r + np.sqrt(
            11664 * self.c**8 * r**2 +
            864 * self.c**6 * (1 - 2 * self.c * z)**3
        )
        term = np.power(term, 1/3)
        radii_proj = -(p * (1 - 2 * self.c * z)) / term +\
            term / (6 * p * self.c**2)
        return radii_proj * 1000  # m -> mm

    def __get_rotation(self, base, surface):
        tank = np.vstack((surface, base)).T
        covar = np.cov(tank)
        u, s, vh = np.linalg.svd(covar)
        return vh

    def z(self, r):
        """
        calculate the z coordinate on the hyprobolic function z = self.c * r^2
        given r
        the input/output unit is mm
        """
        z = self.c * (r/1000)**2
        return z * 1000

    def get_xyz(self, points):
        """
        return shape: (3, n)
        """
        return self.rot @ (points.T - self.base)

    def get_cylinder(self, points):
        """
        points: shape is (n, 3)
        output: shape is (3, n)
        """
        XYZ = self.rot @ (points.T - self.base)
        X, Y, Z = XYZ
        theta = np.arctan2(Y, X)
        radii = np.sqrt(X**2 + Y**2)
        return np.array([radii, theta, Z])

    def get_projection(self, points):
        """
        input shape (n, 3)
        output shape (3, n)
        """
        radii, theta, z = self.get_cylinder(points)
        radii_proj = self.__project_r(radii, z)
        z_proj = self.z(radii_proj)
        proj_cart = np.empty((3, len(z)), dtype=np.float64)
        proj_cart[0] = radii_proj * np.cos(theta)  # x = r * cos(t)
        proj_cart[1] = radii_proj * np.sin(theta)  # y = r * sin(t)
        proj_cart[2] = z_proj
        return (self.rot.T @ proj_cart) + self.base  # (3, n)

    def get_curvilinear(self, points):
        radii, theta, z = self.get_cylinder(points)
        radii_proj = self.__project_r(radii, z)

        c = self.c / 1000
        arc_length = 0.5 * radii_proj * np.sqrt(
                1 + 4 * c**2 * radii_proj**2
                ) + np.arcsinh(2 * c * radii_proj) / (4 * c)

        distance = np.linalg.norm(
            np.vstack((radii, z)) -
            np.vstack((radii_proj, self.z(radii_proj))),
            axis=0
        )
        curvilinear = np.empty((3, len(z)), dtype=np.float64)
        curvilinear[0] = arc_length
        curvilinear[1] = theta
        curvilinear[2] = distance
        return curvilinear

    def random(self, number):
        """
        random points inside tank
        use rejection method to generate random points inside the tank
        """
        succeed = False
        while not succeed:
            x, y, z = np.random.random((3, number * 10)) * np.array(
                    (self.r_max, self.r_max, self.z_max)
                    )
            mask = np.zeros(len(z), dtype=bool)
            r = np.sqrt(x**2 + y**2)
            mask[z > self.z(r)] = True
            if np.sum(mask) > number:
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
        c = self.c / 1000
        for i, l in enumerate(L):
            rp = root_scalar(a2r_cost, args=(c, l), x0=l, x1=l/2).root
            zp = self.z(rp)
            s = 2 * c * rp  # slope
            theta = np.pi/2 - np.arctan(s)
            dr = D[i] * np.cos(theta)
            dz = D[i] * np.sin(theta)
            R[i] = rp - dr
            Z[i] = zp + dz
        return R, Z


def get_nn(positions, no_vertices=True):
    """
    calculate the NN distances and relative locations of NNs
    if no_vertices is True, ignore the vertices
        of the convex hull generated by positoins
    """
    if no_vertices:
        cv = ConvexHull(positions)
        not_vertices = np.array([
            x for x in np.arange(len(cv.points)) if x not in cv.vertices
        ])
        focus = cv.points[not_vertices]
    else:
        focus = positions
    dist_matrix = cdist(focus, positions)
    dist_matrix[dist_matrix == 0] = np.inf
    nn_dists = dist_matrix.min(axis=1)
    nn_indices = dist_matrix.argmin(axis=1)
    nn_locations = positions[nn_indices] - focus
    return nn_locations, nn_dists


def get_nn_with_velocity(positions, velocities, no_vertices=True):
    """
    Calculate the NN distances and relative locations of NNs
    The distances were rotated so that the *x-axis* is aligned
        with the *velocity* of different particles
    If no_vertices is True, ignore the vertices of the convex hull
    """
    if no_vertices:
        cv = ConvexHull(positions)
        not_vertices = np.array([
            x for x in np.arange(len(cv.points)) if x not in cv.vertices
        ], dtype=int)
        if len(not_vertices) == 0:
            return np.empty((0, 3)), np.empty(0)
        focus = cv.points[not_vertices]
    else:
        focus = positions
    dist_matrix = cdist(focus, positions)
    dist_matrix[dist_matrix == 0] = np.inf
    nn_dists = dist_matrix.min(axis=1)
    nn_indices = dist_matrix.argmin(axis=1)
    vx, vy, vz = velocities.T
    azi = np.arctan2(vy, vx)
    ele = np.arctan2(vz, np.sqrt(vx**2 + vy**2))
    rot_vecs = np.vstack((np.zeros(len(vx)), -ele, -azi)).T
    rot_mats = Rotation.from_rotvec(rot_vecs).as_dcm()
    nn_locations = positions[nn_indices] - focus
    nn_locations = np.array([r@n for r, n in zip(rot_mats, nn_locations)])
    return nn_locations, nn_dists


def box_density_polar(
        positions: np.array, centre: np.array, radius: int,
        n_radical: int, n_angular: int
        ):
    """
    Generate mesh in a polar coordinate, each box has the same area
    Calculate the mean & std of number points in each bin

    :param positions: particle positions (n, 2)
    :param centre: boundary centre, shape (2, )
    :param radius: the radius of the boundary
    :param n_radical: number of bins radically
    :param n_angular: number of bins for the angles
    """
    shifted = positions - centre[np.newaxis, :]
    counts = np.ones(shifted.shape[0])
    polar = np.empty(positions.shape)
    polar[:, 0] = np.linalg.norm(shifted, axis=1)
    polar[:, 1] = np.arctan2(*shifted.T) + np.pi  # range (0, 2 pi)
    bins_ang = np.arange(n_angular + 1) / n_angular * np.pi * 2
    #bins_rad = np.sqrt(np.arange(n_radical + 1) / n_radical) * radius
    bins_rad = np.empty(n_radical+1)
    r0 = np.sqrt(radius**2 / (n_angular * (n_radical-1) + 1))
    bins_rad[0], bins_rad[1] = 0, r0
    for i in range(2, n_radical+1):
        bins_rad[i] = np.sqrt(((i-1) * n_angular + 1))* r0
    numbers, _, _, _ = binned_statistic_2d(
        *polar.T, counts, statistic='count', bins=(bins_rad, bins_ang)
    )
    return np.mean(numbers), np.std(numbers)
