#!/usr/bin/env python3
import itertools
import numpy as np
from typing import List, Tuple
from .camera import get_fundamental
from .stereolink import triangulation_v3
# Some helper function I wrote to reconstruct ellipses from three views
# This is to calculate the orientation of my fish tank


def get_conic_coef(a: float, b: float, xc: float, yc: float, rotation: float) -> Tuple[float]:
    """
    Represent an ellipse from geometric form (a, b, x_centre, y_centre, rotation)
    to the algebraic form (α x^2 + β x y + γ y^2 + δ x + ε y + η)
    """
    c, d, t = xc, yc, rotation
    sin, cos = np.sin, np.cos  # lazy bone!
    alpha   = a**2 * sin(t)**2 + b**2 * cos(t) ** 2
    beta    = 2 * (a**2 - b**2) * sin(t) * cos(t)
    gamma   = a**2 * cos(t)**2 + b**2 * sin(t)**2
    delta   = -2 * (d * (a**2 - b**2) * sin(t) * cos(t) + a**2 * c * sin(t)**2 + b**2 * c * cos(t)**2)
    epsilon = -2 * (c * (a**2 - b**2) * sin(t) * cos(t) + a**2 * d * cos(t)**2 + b**2 * d * sin(t)**2)
    eta     = sin(t)**2 * (a**2 * c**2 + b**2 * d**2) + cos(t)**2 * (a**2 * d**2 + b**2 * c**2) +\
              2 * c * d * (a**2 - b**2) * sin(t) * cos(t) - a**2 * b**2
    return (alpha, beta, gamma, delta, epsilon, eta)


def parse_ellipses_imagej(csv_file):
    """
    the ellipse expression corresponds to (u, v) coordinate system
    """
    ellipse_pars = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=[1,2,3,4,5])
    xc, yc, major, minor, rotation = ellipse_pars.T
    rotation = rotation / 180 * np.pi
    a = major / 2
    b = minor / 2
    return np.vstack((a, b, xc, yc, rotation)).T


def draw_ellipse(angle: np.ndarray, ellipse: List[float]) -> np.ndarray:
    """
    return (u, v) coordinates for plotting
    no need to use `np.flip` to plot with figure
    """
    a, b, xc, yc, rotation = ellipse
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
    al, be, xc, yc, t = ellipse
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
    find the intersection between an ellipse and a line
    :param ellipse: represented as (a, b, xc, yc, rot), the geometrical form
    :param line:    represented as (l1, l2, l3) where l1 x + l2 y + l3 == 0
    * If there is no intersection, then find the point on the ellipse that is closest to the line.
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


def match_ellipse_sloopy(cameras: List['Camera'], ellipses: List[List[float]], N: int, min_diff=250):
    """
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
        if costs[1] - costs[0] < 250:
            continue
        p2h, p3h = pairs_homo[chosen]
        points.append(triangulation_v3((p1h, p2h, p3h), cameras))
    return np.array(points)
