#!/usr/bin/env python3
from . import tower_sample as ts
import warnings
from itertools import product
import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.spatial import ConvexHull
from scipy.special import gamma


def auto_corr(var, dt=1):
    fluctuation = np.array(var) - np.nanmean(var)
    stop = len(var)
    corr = fluctuation[dt:stop] * fluctuation[:stop-dt]
    c = np.nanmean(corr)
    std = np.nanstd(corr)
    c0 = np.nanmean(fluctuation * fluctuation)
    return c/c0, std/c0


def get_acf(var):
    """
    Calculate the auto-correlation function for a 1D variable
    """
    size = len(var)
    result = np.empty(size - 1, dtype=np.float64)
    variations = np.empty(size - 1, dtype=np.float64)
    for dt in range(0, size - 1):
        acf, std = auto_corr(var, dt)
        result[dt] = acf
        variations[dt] = std
    return result, variations


def get_centre(trajectories, frame):
    positions = []
    for t in trajectories:
        if frame in t.time:
            index = np.where(t.time == frame)[0][0]
            positions.append(t.positions[index])
    if len(positions) > 0:
        return np.mean(positions, 0)
    else:
        warn(f"No positions found in frame {frame}")


def get_centre_move(trajectories, frame):
    """
    calculate the movement of the centre from [frame] to [frame + 1]
    only the trajectories who has foodsteps in both frames were used
    """
    movements = []
    for t in trajectories:
        if (frame in t.time) and (frame + 1 in t.time):
            index_1 = np.where(t.time == frame)[0][0]
            index_2 = np.where(t.time == frame + 1)[0][0]
            movements.append(t.positions[index_2] - t.positions[index_1])
    return np.mean(movements, axis=0)


def get_centres(trajectories, frames):
    centres = np.empty((len(frames), 3), dtype=np.float64)
    for i, frame in enumerate(frames):
        centres[i] = get_centre(trajectories, frame)
    return centres


def get_best_rotation(r1, r2):
    """
    calculate the best rotation to relate two sets of vectors
    see the paper [A solution for the best rotation to relate two sets of vectors] for detail
    all the points were treated equally, which means w_n = 0 (in the paper)
    """
    R = r2.T @ r1
    u, a = np.linalg.eig(R.T @ R)  # a[:, i] corresponds to u[i]
    b = 1 / np.sqrt(u) * (R @ a)
    return (b @ a.T).T


def get_best_dilatation_rotation(r1, r2, init_guess=None):
    """
    calculate numeratically
    (r1 @ Lambda) @ Rotation = r2, hopefully
    """
    if isinstance(init_guess, type(None)):
        init_guess = np.ones(r1.shape[1])
    def cost(L, r1, r2):
        Lambda = np.identity(r1.shape[1]) * np.array(L)
        r1t = r1 @ Lambda
        R = get_best_rotation(r1t, r2)
        return np.sum(np.linalg.norm(r2 - r1t @ R))
    result = least_squares(cost, init_guess, args=(r1, r2))
    L = np.identity(r1.shape[1]) * np.array(result['x'])
    r1t = r1 @ L
    R = get_best_rotation(r1t, r2)
    return L, R


def get_convex_hull(trajectories, target_num=0):
    frames = list(set(np.hstack([t.time for t in trajectories]).ravel()))
    for frame in frames:
        points = []
        for t in trajectories:
            if frame in t.time:
                points.append(t.positions[t.time == frame])
        points = np.squeeze(points)
        if len(points) >= target_num:
            yield ConvexHull(np.array(points))


def get_rg_tensor(trajectories, target_num=0):
    frames = list(set(np.hstack([t.time for t in trajectories]).ravel()))
    for frame in frames:
        points = []
        for t in trajectories:
            if frame in t.time:
                points.append(t.positions[t.time == frame])
        points = np.squeeze(points)
        if len(points) >= target_num:
            yield np.cov((points - points.mean(0)).T)


class GCE:
    def __init__(self, trajs, good_frames=None):
        """
        Estimating the Group Centre, trying to use knowledge of good frames to reduce the error
        param: trajs:
            a list/tuple of many [trajectory] objects
        param: trajectory:
            a dict with {'positions': (frames, dimension) numpy array, 'time': (frames,) numpy array}
        param: good_frames:
            *frame number* that the group centre can be well estimated
        """
        self.trajs = trajs
        self.frames = list(set(np.hstack([t.time for t in self.trajs]).ravel()))
        if len(self.frames) != np.max(self.frames)+1:
            warnings.warn("Missing some frame in all the trajectoreis")
        self.centres = np.zeros((len(self.frames), 3))

        if isinstance(good_frames, type(None)):
            self.centres = get_centres(self.trajs, self.frames)
        elif isinstance(good_frames, int):
            self.good_frames = self.__detect(good_frames)
            self.__diffuse()
        else:
            self.good_frames = good_frames
            self.__diffuse()

    def __get_traj_nums(self):
        """
        count the number of trajectories at each frame
        """
        numbers = np.zeros(len(self.frames), np.int64)
        for i, frame in enumerate(self.frames):
            for traj in self.trajs:
                if frame in traj.time:
                    numbers[i] += 1
        return numbers

    def __detect(self, target_num):
        numbers = self.__get_traj_nums()
        return np.array(self.frames)[numbers >= target_num]

    def __diffuse(self):
        """
        expand the good frames so entire frames is devided to different regions
        inside each region, the centres is calculated by summing δ(centres) to reduce the error

        say i, j, k are good frames
        -             frames < (j+i)//2 --> i
        - (j+i)//2 <= frames < (j+k)//2 --> j
        - (j+k)//2 <= frames            --> k

        here are the graphical illustration, dashed lines are boundaries

        ┊<-- i ->┊<-- j --->┊<--- k -->┊
        ┊        ┊          ┊          ┊
        ┊────┴───┊────┴─────┊─────┴────┊
        """
        boundaries = [0]
        boundaries += [(i+j)//2 for (i, j) in zip(self.good_frames[:-1], self.good_frames[1:])]
        boundaries.append(np.max(self.frames))
        ranges = zip(boundaries[:-1], boundaries[1:])
        for gf, r in zip(self.good_frames, ranges):
            gi = self.frames.index(gf)  # good index
            source = get_centre(self.trajs, gf)  # other centres were diffused from the source

            self.centres[gi] = source

            left = source.copy()  # reversing time, going left <--
            for i, frame in enumerate(range(gf-1, r[0]-1, -1)):
                left -= get_centre_move(self.trajs, frame)
                self.centres[gi-i-1] = left

            right = source.copy()
            for i, frame in enumerate(range(gf, r[1], +1)):
                right += get_centre_move(self.trajs, frame)
                self.centres[gi+i+1] = right


def maxwell_boltzmann_nd(v, theta, v_sq_mean):
    alpha = (1 + theta) / 2
    lambda_ = v_sq_mean * gamma(alpha) / gamma(alpha + 1)
    term_1 = 2 * v ** theta
    term_2 = gamma(alpha) * lambda_ ** alpha
    term_3 = np.exp(-v ** 2 / lambda_)
    return term_1 / term_2 * term_3


def fit_maxwell_boltzmann(speed, bins):
    """
    fit the speed distribution with maxwell boltzmann distribution
    the average of the energy (< speed ** 2> is fixed)
    return: the dimension & fitting function
    """
    spd_sqr_mean = np.nanmean(speed ** 2)
    spd_pdf, bin_edges = np.histogram(speed, bins=bins, density=True)
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

    res = curve_fit(
        lambda x, theta: maxwell_boltzmann_nd(x, theta, spd_sqr_mean),
        bin_centres,
        spd_pdf, (2.0),  # start with initial guess of 3D world
    )

    dimension = res[0][0] + 1

    fit_x = np.linspace(0, bins[-1], 200)
    fit_y = maxwell_boltzmann_nd(fit_x, dimension-1, spd_sqr_mean)

    return dimension, (bin_centres, spd_pdf), (fit_x, fit_y)


def biased_discrete_nd(variables, bins, size=1):
    """
    Generate random numbers that have the joint distribution of many variables
    Using tower sampling
    :param variabels:
        a iterable of random variables, each variable is a numpy array
        shape: (Number, Dimension)
    :param bins:
        the bin for generating the discrete distribution, it can be either
        shape: (Dimension, bin_number)
    :param size:
        the number of generated random numbers
    """
    discrete_pdf, bin_edges = np.histogramdd(variables, bins=bins)
    bin_centres = np.array([(be[1:] + be[:-1]) / 2 for be in bin_edges])
    maps = np.array(list(product(*bin_centres)))# map from bin indices to values in the bin
    random_indices = ts.tower_sampling(size, discrete_pdf.astype(np.int64))
    random_numbers = maps[random_indices]
    return random_numbers
