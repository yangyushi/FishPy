#!/usr/bin/env python3
"""
Some helper function I wrote to reconstruct ellipses from three views
This is to calculate the orientation of my fish tank
"""
import itertools
import numpy as np
from typing import List, Tuple
from .camera import get_fundamental
from .stereolink import triangulation_v3
from scipy.spatial.transform import Rotation


def fit_ellipse(data):
    """
    Estimate circle model from data using total least squares.

    I copied it from the scikit-image library

    References: Halir, R.; Flusser, J. "Numerically stable direct\
        least squares fitting of ellipses"

    Args:
        data (numpy.ndarray): 2d features, shape (n, 2)

    Return:
        numpy.ndarray: the coefficients of the conic in the form of\
            :math:`c_1 x^2 + c_2 xy + c^3 y^2 + c^4 x + c^5 y + c_6 = 0`
    """
    # to prevent integer overflow
    float_type = np.promote_types(data.dtype, np.float32)
    x, y = data.astype(float_type).T

    # Quadratic part of design matrix [eqn. 15] from [1]
    D1 = np.vstack([x ** 2, x * y, y ** 2]).T

    # Linear part of design matrix [eqn. 16] from [1]
    D2 = np.vstack([x, y, np.ones_like(x)]).T

    # forming scatter matrix [eqn. 17]
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # Constraint matrix [eqn. 18]
    C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

    try:
        M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)  # [eqn. 29]
    except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
        return False

    eig_vals, eig_vecs = np.linalg.eig(M) # [eqn. 28]

    # eigenvector must meet constraint 4ac - b^2 to be valid.
    cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) \
           - np.power(eig_vecs[1, :], 2)
    a1 = eig_vecs[:, (cond > 0)]

    # seeks for empty matrix
    if 0 in a1.shape or len(a1.ravel()) != 3:
        return False
    a, b, c = a1.ravel()

    # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
    a2 = -np.linalg.inv(S3) @ S2.T @ a1
    d, f, g = a2.ravel()

    # eigenvectors are the coefficients of an ellipse in general form
    # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
    b /= 2.
    d /= 2.
    f /= 2.

    # finding center of ellipse
    x0 = (c * d - b * f) / (b ** 2. - a * c)  # b = B*2
    y0 = (a * f - b * d) / (b ** 2. - a * c)

    # Find the semi-axes lengths
    numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 \
                - 2 * b * d * f - a * c * g
    term = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    denominator1 = (b ** 2 - a * c) * (term - (a + c))
    denominator2 = (b ** 2 - a * c) * (- term - (a + c))
    width = np.sqrt(2 * numerator / denominator1)
    height = np.sqrt(2 * numerator / denominator2)

    phi = 0.5 * np.arctan((2. * b) / (a - c))
    if a > c:
        phi += 0.5 * np.pi

    params = np.nan_to_num([x0, y0, width, height, phi]).tolist()
    params = [float(np.real(x)) for x in params]
    return get_conic_coef(*params)


    return coef


def get_conic_coef(xc, yc, a, b, rotation):
    """
    Represent an ellipse from geometric form (a, b, x_centre, y_centre, rotation)
    to the algebraic form (α x^2 + β x y + γ y^2 + δ x + ε y + η)

    The parameters are consistense with the parameters of skimage.measure.EllipseModel

    Args:
        xc (float): x_centre
        yc (float): y_centre
        a (float): a
        b (float): b
        rotation (float): rotation

    Return:
        tuple: (α, β, γ, δ, ε, η)
    """
    st, ct = np.sin(rotation), np.cos(rotation)
    alpha = st**2/b**2 + ct**2/a**2
    beta = - 2*st*ct/b**2 + 2*st*ct/a**2
    gamma = ct**2/b**2 + st**2/a**2
    delta = -2*xc*st**2/b**2 + 2*yc*st*ct/b**2 - 2*xc*ct**2/a**2 - 2*yc*st*ct/a**2
    epsilon = 2*xc*st*ct/b**2 - 2*yc*ct**2/b**2 - 2*xc*st*ct/a**2 - 2*yc*st**2/a**2
    eta = -1 + xc**2*st**2/b**2 - 2*xc*yc*st*ct/b**2 + yc**2*ct**2/b**2
    eta += xc**2*ct**2/a**2 + 2*xc*yc*st*ct/a**2 + yc**2*st**2/a**2
    return alpha, beta, gamma, delta, epsilon, eta


def get_conic_matrix(conic_coef):
    """
    Assemble the conic coefficients into a conic matrix (AQ)

    See the `wiki page <https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections>`_

    Args:
        conic_coef (iterable): a container with 6 numbers

    Return:
        np.ndarray: the conic matrix
    """
    A, B, C, D, E, F = conic_coef
    matrix = (
        (A, B/2, D/2),
        (B/2, C, E/2),
        (D/2, E/2, F)
    )
    return np.array(matrix)


def get_geometric_coef(matrix):
    r"""
    Calculate the geometric coefficients of the conic from a conic Matrix:

    $$
    \\begin{array}{c}
    A & B/2 & D/2 \\\\
    B/2 &   C & E/2 \\\\
    D/2 & E/2 &   F
    \\end{array}
    $$

    The conic form is written as,

    $$
    A x^2 + B xy + C y^2 + D x + E y + F = 0
    $$

    Reference: `Wikipedia <https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections>`_


    Args:
        matrix (np.ndarray): the conic matrix, being AQ in the wiki.

    Return:
        tuple: (xc, yc, a, b, rotation)
    """
    triu_idx = np.triu_indices_from(matrix)
    A, B, D, C, E, F = matrix[triu_idx]
    B, D, E = map(lambda x: 2 * x, (B, D, E))
    xc = (B*E - 2*C*D) / (4*A*C - B**2)
    yc = (D*B - 2*A*E) / (4*A*C - B**2)
    A33 = np.array((
        (A, B/2),
        (B/2, C)
    ))
    (lambda_1, lambda_2), rot_mat = np.linalg.eigh(A33)
    rot_mat = rot_mat / np.linalg.norm(rot_mat, axis=0)
    K = - np.linalg.det(matrix) / np.linalg.det(A33)
    theta = 0.5 * np.arctan2(B, A-C)
    if K < 0:
        return xc, yc, np.nan, np.nan, theta
    else:
        a = np.sqrt(K / lambda_2)
        b = np.sqrt(K / lambda_1)
        return xc, yc, a, b, theta


def parse_ellipses_imagej(csv_file):
    """
    the ellipse expression corresponds to (u, v) coordinate system
    """
    ellipse_pars = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=[1,2,3,4,5])
    xc, yc, major, minor, rotation = ellipse_pars.T
    rotation = rotation / 180 * np.pi
    a = major / 2
    b = minor / 2
    return np.vstack((xc, yc, a, b, rotation)).T


def draw_ellipse(angle: np.ndarray, ellipse: List[float]) -> np.ndarray:
    """
    return (u, v) coordinates for plotting
    no need to use `np.flip` to plot with figure
    """
    xc, yc, a, b, rotation = ellipse
    rotation *= -1
    x_eb = np.cos(angle) * a  # ellipse basis
    y_eb = np.sin(angle) * b
    rot_mat = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)],
    ])
    shift = np.vstack((xc, yc))
    # tranform from "ellipse basis" to "standard basis" 
    return rot_mat.T @ np.vstack((x_eb, y_eb)) + shift


def find_projection(ellipse, line):
    """
    find the point on an ellipse that is closest to a line
    :param ellipse: represented as (al, bl, xc, yc, rot), the geometrical form
    :param line:    represented as (a1, a2, a3) where l1 x + l2 y + l3 == 0
    I am sorry for the inconsistent notations
    """
    Sqrt = np.sqrt
    xc, yc, al, be, t = ellipse
    a1, b1, c1 = line
    a = a1 * np.cos(t) + b1 * np.sin(t)
    b = b1 * np.cos(t) - a1 * np.sin(t)
    c = c1 - a1 * xc * np.cos(t) - b1 * yc * np.cos(t) -\
        b1 * xc * np.sin(t) + a1 * yc * np.sin(t)
    # calculating in "ellipse basis", results copied from mathematica
    x1 = -((a*al**2)/Sqrt(a**2*al**2 + b**2*be**2))
    y1 = -((b*be**2)/Sqrt(a**2*al**2 + b**2*be**2))
    x2 = (a*al**2)/Sqrt(a**2*al**2 + b**2*be**2)
    y2 = (b*be**2)/Sqrt(a**2*al**2 + b**2*be**2)
    extrema = np.array(((x1, x2), (y1, y2))) # shape (2, 2)
    dist_sq = (a * extrema[0] + b * extrema[1] + c) **2 / (a**2 + b**2)
    index = np.argmin(dist_sq)
    projection_eb = extrema.T[1]  # ellipse basis
    # converting back to "standard basis"
    rot_mat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    x, y = rot_mat.T @ projection_eb + np.array([xc, yc])
    return np.array([[x], [y]])


def get_intersection(ellipse, line):
    """
    Finding the intersection between an ellipse and a line
    * If there is no intersection, then find the point on the ellipse that is closest to the line.

    Args:
        ellipse (:obj:`numpy.ndarray`): represented as (a, b, xc, yc, rot), the geometrical form
        line (:obj:`numpy.ndarray`):    represented as (l1, l2, l3) where l1 x + l2 y + l3 == 0

    """
    al, be, ga, de, ep, et = get_conic_coef(*ellipse)
    l1, l2, l3 = line
    a, b = -(l1 / l2), -(l3 / l2)
    term_1 = (b * (be + 2 * a * ga) + de + a * ep) ** 2
    term_2 = -4 * (al + a * (be + a * ga))
    term_3 = (b**2 * ga + b * ep + et)
    if term_1 + (term_2 * term_3) < 0:
        return find_projection(ellipse, line)  # degenrate case 1 where line is not crossing ellipse
    elif abs(term_1 + (term_2 * term_3)) < 1e-15:
        term_sq = 0  # degenrate case 2 where there is only one intersection point
        denominator = -2 * (al + a * (be + a * ga))
        x = term_x / denominator
        y = -1 * term_y / denominator
        return np.array([[x1, x2]])
    else:
        term_sq = np.sqrt(term_1 + (term_2 * term_3))
        denominator = -2 * (al + a * (be + a * ga))
        term_x = b * be + 2 * a * b * ga + de + a * ep
        term_y = 2 * b * al + a * b * be - a * de - a**2 * ep
        x1 = (term_x + term_sq) / denominator
        x2 = (term_x - term_sq) / denominator
        y1 = (-1 * term_y + a * term_sq) / denominator
        y2 = (-1 * term_y - a * term_sq) / denominator
        return np.array([[x1, x2], [y1, y2]])


def match_ellipse_sloopy(
    cameras: List['Camera'], ellipses: List[List[float]], N: int,
    min_diff=250, max_cost=10
):
    """
    .. code-block::

        1. Randomly choose N points in view 1
        3. For every chosen point (P1)
            1. Calculate POI of P1 in other two views, get POI2 & POI3
            2. For POI2, calculate its projection on C2, get P2 (using camera's information)
            3. For POI3, calculate its projection on C3, get P3
            4. Reconstruct 3D point using P1, P2, & P3
        4. These points should be points on the surface
    """
    points_v1 = draw_ellipse(np.linspace(0, 2*np.pi, N, endpoint=False), ellipses[0])  # (u, v)
    points = []
    f12 = get_fundamental( cameras[0], cameras[1] )
    f13 = get_fundamental( cameras[0], cameras[2] )
    f23 = get_fundamental( cameras[1], cameras[2] )
    f12 = f12 / f12.max()
    f13 = f13 / f13.max()
    f23 = f23 / f23.max()
    for i, p1 in enumerate(points_v1.T):
        p1h = np.array((*p1, 1))  # homogeneous
        line_2 = f12 @ p1h
        line_3 = f13 @ p1h
        cross_12 = get_intersection(ellipses[1], line_2).T  # [(x1, y1) & (x2, y2)]
        cross_13 = get_intersection(ellipses[2], line_3).T
        if (len(cross_12) == 1) or (len(cross_13) == 1):
            continue
        pairs = list(itertools.product(cross_12, cross_13))
        costs, pairs_homo = [], []
        for p2, p3 in pairs:
            p2h = np.array((*p2, 1))  # homogeneous
            p3h = np.array((*p3, 1))
            pairs_homo.append((p2h, p3h))
            costs.append(
                np.abs(p2h @ f23.T @ p3h) + np.abs(p1h @ f13.T @ p3h) + np.abs(p1h @ f12.T @ p2h)
            )
        chosen = np.argmin(costs)
        costs.sort()
        if costs[1] - costs[0] < min_diff:
            continue
        if costs[0] < max_cost:
            p2h, p3h = pairs_homo[chosen]
            points.append(triangulation_v3((p1h, p2h, p3h), cameras))
    return np.array(points)


def cost_conic(RT, K, C, p2d, p3dh):
    """
    Cost function to incorporate the conic measurement into camera calibration.
    It is aimed to optimise the result of the cv2.solvePnP function.
    The conic in the image should correspond to a circle in 3D @ plane Z=0.

    Args:
        RT (np.ndarray): the rotation and translation to be optimised.\
            shape (6, )
        K (np.ndarray): the intrinsic camera matrix, shape (3, 3)
        C (np.ndarray): the matrix for the measured conic, shape (3, 3)
        p2d (np.ndarray): 2d features for the solvePnP method, shape (n, 2)
        p3dh (np.ndarray): the *homogeneous* representation of \
            3d positions for the solvePnP method, shape (n, 4)

    Return:
        float: the geometric distance
    """
    R_mat = Rotation.from_rotvec(RT[:3]).as_matrix()  # (3, 3)
    T = np.array(RT[3:])[:, np.newaxis]  # (3, 1)
    P = np.concatenate((R_mat, T), axis=1)  # (3, 4)
    P = K @ P
    Q = P.T @ C @ P

    p2d_proj = P @ p3dh.T
    p2d_proj /= p2d_proj[-1]
    p2d_proj = p2d_proj[:2].T

    conic_coef = (
        Q[0, 0], Q[0, 1] + Q[1, 0], Q[1, 1],
        Q[0, 3] + Q[3, 0], Q[1, 3] + Q[3, 1],
        Q[3, 3]
    )

    conic_mat = get_conic_matrix(conic_coef)
    xc, yc, a, b, _ = get_geometric_coef(conic_mat)
    # ensure the original PnP constrain
    cost_val = np.sum((p2d - p2d_proj) ** 2) / len(p3dh)
    # make sure the reconstructed circlej
    cost_val += (a - b) ** 2
    return cost_val


def cost_circle_triple(
        RT123, K1, K2, K3, C1, C2, C3, p2d1, p2d2, p2d3, p3dh
):
    """
    Cost function to incorporate the conic measurement into camera calibration.
    It is aimed to optimise the result of the cv2.solvePnP function.
    The conic in the image should correspond to a circle in 3D @ plane Z=0.

    Args:
        RT123 (numpy.ndarray): the rotation and translation to be optimised,\
            shape (18,)
        K1 (numpy.ndarray): the calibration matrix\
            :math:`\\in \\mathbb{R}^{3 \\times 3}` of camera 1
        K2 (numpy.ndarray): the calibration matrix\
            :math:`\\in \\mathbb{R}^{3 \\times 3}` of camera 2
        K3 (numpy.ndarray): the calibration matrix\
            :math:`\\in \\mathbb{R}^{3 \\times 3}` of camera 3
        C1 (numpy.ndarray): the conic matrix from camera 1, shape (3, 3)
        C2 (numpy.ndarray): the conic matrix from camera 2, shape (3, 3)
        C3 (numpy.ndarray): the conic matrix from camera 3, shape (3, 3)
        p2d1 (numpy.ndarray): the 2d features for the `solvePnP` method\
            from camera 1, shape (n, 2)
        p2d2 (numpy.ndarray): the 2d features for the `solvePnP` method\
            from camera 2, shape (n, 2)
        p2d3 (numpy.ndarray): the 2d features for the `solvePnP` method\
            from camera 3, shape (n, 2)
        p3dh (numpy.ndarray): the *homogeneous* representation of \
            3d positions for the solvePnP method, shape (n, 4)

    Return:
        float: the geometric distance
    """

    R1 = Rotation.from_rotvec(RT123[:3])
    T1 = np.array(RT123[3:6])
    R2 = Rotation.from_rotvec(RT123[6:9])
    T2 = np.array(RT123[9:12])
    R3 = Rotation.from_rotvec(RT123[12:15])
    T3 = np.array(RT123[15:18])

    P1 = K1 @ np.concatenate((R1.as_matrix(), T1[:, None]), axis=1)
    P2 = K2 @ np.concatenate((R2.as_matrix(), T2[:, None]), axis=1)
    P3 = K3 @ np.concatenate((R3.as_matrix(), T3[:, None]), axis=1)

    Q1 = P1.T @ C1 @ P1
    Q2 = P2.T @ C2 @ P2
    Q3 = P3.T @ C3 @ P3

    p2d_proj_1 = P1 @ p3dh.T
    p2d_proj_1 /= p2d_proj_1[-1]
    p2d_proj_1 = p2d_proj_1[:2].T

    p2d_proj_2 = P2 @ p3dh.T
    p2d_proj_2 /= p2d_proj_2[-1]
    p2d_proj_2 = p2d_proj_2[:2].T

    p2d_proj_3 = P3 @ p3dh.T
    p2d_proj_3 /= p2d_proj_3[-1]
    p2d_proj_3 = p2d_proj_3[:2].T

    xc_1, yc_1, a_1, b_1, _ = get_geometric_coef(get_conic_matrix((
        Q1[0, 0], 2 * Q1[0, 1], Q1[1, 1], 2 * Q1[0, 3], 2 * Q1[1, 3], Q1[3, 3]
    )))
    xc_2, yc_2, a_2, b_2, _ = get_geometric_coef(get_conic_matrix((
        Q2[0, 0], 2 * Q2[0, 1], Q2[1, 1], 2 * Q2[0, 3], 2 * Q2[1, 3], Q2[3, 3]
    )))
    xc_3, yc_3, a_3, b_3, _ = get_geometric_coef(get_conic_matrix((
        Q3[0, 0], 2 * Q3[0, 1], Q3[1, 1], 2 * Q3[0, 3], 2 * Q3[1, 3], Q3[3, 3]
    )))


    cost_val_pnp = np.sum((
        np.sum((p2d1 - p2d_proj_1) ** 2),
        np.sum((p2d2 - p2d_proj_2) ** 2),
        np.sum((p2d3 - p2d_proj_3) ** 2),
    ))

    cost_val_centre = np.sum((
        (xc_1 - xc_2) ** 2, (xc_1 - xc_3) ** 2,
        (yc_1 - yc_2) ** 2, (yc_1 - yc_3) ** 2,
    ))

    cost_val_circle = np.sum((
        (a_1 - a_2) **2, (a_1 - a_3) **2,
        (a_1 - b_1) **2, (a_2 - b_2) **2, (a_3 - b_3) **2,
    ))

    return cost_val_pnp / len(p3dh) + cost_val_centre + cost_val_circle


def cost_conic_triple(
        RT123, K1, K2, K3, C1, C2, C3, p2d1, p2d2, p2d3, p3dh
):
    """
    Cost function to incorporate the conic measurement into camera calibration.
    It is aimed to optimise the result of the cv2.solvePnP function.
    The conic in the image should correspond to a circle in 3D @ plane Z=0.

    Args:
        RT123 (numpy.ndarray): the rotation and translation to be optimised,\
            shape (18,)
        K1 (numpy.ndarray): the calibration matrix\
            :math:`\\in \\mathbb{R}^{3 \\times 3}` of camera 1
        K2 (numpy.ndarray): the calibration matrix\
            :math:`\\in \\mathbb{R}^{3 \\times 3}` of camera 2
        K3 (numpy.ndarray): the calibration matrix\
            :math:`\\in \\mathbb{R}^{3 \\times 3}` of camera 3
        C1 (numpy.ndarray): the conic matrix from camera 1, shape (3, 3)
        C2 (numpy.ndarray): the conic matrix from camera 2, shape (3, 3)
        C3 (numpy.ndarray): the conic matrix from camera 3, shape (3, 3)
        p2d1 (numpy.ndarray): the 2d features for the `solvePnP` method\
            from camera 1, shape (n, 2)
        p2d2 (numpy.ndarray): the 2d features for the `solvePnP` method\
            from camera 2, shape (n, 2)
        p2d3 (numpy.ndarray): the 2d features for the `solvePnP` method\
            from camera 3, shape (n, 2)
        p3dh (numpy.ndarray): the *homogeneous* representation of \
            3d positions for the solvePnP method, shape (n, 4)

    Return:
        float: the geometric distance
    """

    R1 = Rotation.from_rotvec(RT123[:3])
    T1 = np.array(RT123[3:6])
    R2 = Rotation.from_rotvec(RT123[6:9])
    T2 = np.array(RT123[9:12])
    R3 = Rotation.from_rotvec(RT123[12:15])
    T3 = np.array(RT123[15:18])

    P1 = K1 @ np.concatenate((R1.as_matrix(), T1[:, None]), axis=1)
    P2 = K2 @ np.concatenate((R2.as_matrix(), T2[:, None]), axis=1)
    P3 = K3 @ np.concatenate((R3.as_matrix(), T3[:, None]), axis=1)

    Q1 = P1.T @ C1 @ P1
    Q2 = P2.T @ C2 @ P2
    Q3 = P3.T @ C3 @ P3

    p2d_proj_1 = P1 @ p3dh.T
    p2d_proj_1 /= p2d_proj_1[-1]
    p2d_proj_1 = p2d_proj_1[:2].T

    p2d_proj_2 = P2 @ p3dh.T
    p2d_proj_2 /= p2d_proj_2[-1]
    p2d_proj_2 = p2d_proj_2[:2].T

    p2d_proj_3 = P3 @ p3dh.T
    p2d_proj_3 /= p2d_proj_3[-1]
    p2d_proj_3 = p2d_proj_3[:2].T

    xc_1, yc_1, a_1, b_1, _ = get_geometric_coef(get_conic_matrix((
        Q1[0, 0], 2 * Q1[0, 1], Q1[1, 1], 2 * Q1[0, 3], 2 * Q1[1, 3], Q1[3, 3]
    )))
    xc_2, yc_2, a_2, b_2, _ = get_geometric_coef(get_conic_matrix((
        Q2[0, 0], 2 * Q2[0, 1], Q2[1, 1], 2 * Q2[0, 3], 2 * Q2[1, 3], Q2[3, 3]
    )))
    xc_3, yc_3, a_3, b_3, _ = get_geometric_coef(get_conic_matrix((
        Q3[0, 0], 2 * Q3[0, 1], Q3[1, 1], 2 * Q3[0, 3], 2 * Q3[1, 3], Q3[3, 3]
    )))


    cost_val_pnp = np.sum((
        np.sum((p2d1 - p2d_proj_1) ** 2),
        np.sum((p2d2 - p2d_proj_2) ** 2),
        np.sum((p2d3 - p2d_proj_3) ** 2),
    ))

    cost_val_centre = np.sum((
        (xc_1 - xc_2) ** 2, (xc_1 - xc_3) ** 2,
        (yc_1 - yc_2) ** 2, (yc_1 - yc_3) ** 2,
    ))

    cost_val_ellipse = np.sum((
        (a_1 - a_2) **2, (a_1 - a_3) **2,
    ))

    return cost_val_pnp / len(p3dh) + cost_val_centre + cost_val_ellipse
