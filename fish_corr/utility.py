#!/usr/bin/env python3
from . import tower_sample as ts
import warnings
from itertools import product
import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy.spatial import ConvexHull
from scipy.special import gamma
from scipy.spatial.distance import pdist
from numba import njit


@njit
def get_acf(var, size=0, step=1):
    r"""
    Calculate the auto-correlation function for a n-dimensional variable

    .. math::

        Y[\tau] = \left\langle \sum_i^\text{ndim}{X\left[t, i\right] \cdot X\left[t+\tau, i\right]} \right\rangle

    Args:
        var (:obj:`numpy.ndarray`): a continual nD variable.  the shape is (number, dimension)
        size (:obj:`int`): the maximum size of :math:`\tau`, by default :math:`\tau` == len(var)
        step (:obj:`int`): every ``step`` points in time were chosen as t0 in the calculation

    Return:
        :obj:`numpy.ndarray`: The auto-correlation function of the variable
    """
    length = len(var)
    if size == 0:
        size = length
    dim = var.shape[1]
    mean = np.empty((1, dim), dtype=np.float64)
    for d in range(dim):
        mean[0, d] = np.nanmean(var[:, d])
    flctn = var - mean  # (n, dim) - (1, dim)
    result = np.empty(size, dtype=np.float64)
    for dt in range(0, size):
        stop = length - dt
        corr = np.sum(flctn[: stop : step] * flctn[dt : stop + dt : step], axis=1)
        c0   = np.sum(flctn[: stop : step] * flctn[: stop : step], axis=1)  # normalisation factor
        result[dt] = np.nansum(corr) / np.nansum(c0)
    return result


def get_acf_fft(var, size, nstep, nt):
    """
    not finished
    """
    fft_len = 2 * size # Actual length of FFT data

    # Prepare data for FFT
    fft_inp = np.zeros(fft_len, dtype=np.complex_)
    fft_inp[0:nstep] = v

    fft_out = np.fft.fft(fft_inp) # Forward FFT
    fft_out = fft_out * np.conj ( fft_out ) # Square modulus
    fft_inp = np.fft.ifft(fft_out) # Backward FFT (the factor of 1/fft_len is built in)
    # Normalization factors associated with number of time origins
    n = np.linspace(nstep, nstep - nt, nt + 1, dtype=np.float_)
    assert np.all(n > 0.5), 'Normalization array error' # Should never happen
    c_fft = fft_inp[0 : nt + 1].real / n
    return


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
    calculate the movement of the centre from ``[frame]`` to ``[frame + 1]``
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

    .. code-block::

        (r1 @ Lambda) @ Rotation = r2
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


def get_convex_hull_from_trajs(trajectories, target_num=0):
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
    """
    Estimating the Group Centre, trying to use knowledge of good frames to reduce the error
    """
    def __init__(self, trajs, good_frames=None):
        """
        Args:
            trajs (:obj:`list` or :obj:`tuple`): a list/tuple of many Trajectory objects
            trajectory: content ``{'positions': (frames, dimension) [:obj:`numpy.ndarray`], 'time': (frames,) [:obj:`numpy.ndarray`]}``
            good_frames: frame number that the group centre can be well estimated
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

        .. code-block::

            -             frames < (j+i)//2 --> i
            - (j+i)//2 <= frames < (j+k)//2 --> j
            - (j+k)//2 <= frames            --> k

        here are the graphical illustration, dashed lines are boundaries

        .. code-block::


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

    Args:
        variabels (:obj:`numpy.ndarray`):
            a iterable of random variables, each variable is a numpy array
            shape: (Number, Dimension)
        bins (:obj:`numpy.ndarray`):
            the bin for generating the discrete distribution, it can be either
            shape: (Dimension, bin_number)
        size (:obj:`int`): the number of generated random numbers
    """
    discrete_pdf, bin_edges = np.histogramdd(variables, bins=bins)
    bin_centres = np.array([(be[1:] + be[:-1]) / 2 for be in bin_edges])
    bin_width = np.array([b[1] - b[0] for b in bins]).reshape(1, len(bins))
    maps = np.array(list(product(*bin_centres)))# map from bin indices to values in the bin
    random_indices = ts.tower_sampling(size, discrete_pdf.astype(np.int64))
    random_numbers = maps[random_indices]
    return random_numbers + np.random.random(random_numbers.shape) * bin_width


def get_gr(frames, bins, random_gas):
    """
    Args:
        frames (:obj:`numpy.ndarray`): positions of particles in different frames, shape (frame, n, dim)
        bins (:obj:`numpy.ndarray` or int): the bins for the distance histogram
        random_gas (:obj:`numpy.ndarray`): shape (N, 3)
    """
    offset = 0
    distances = []
    distances_gas = []
    for frame in frames:
        if len(frame) > 2:
            dist = pdist(frame)
            dist_gas = pdist(random_gas[offset : offset+len(frame)] + 1)
            offset += len(frame)
            distances.append(dist)
            distances_gas.append(dist_gas)
    hist, _ = np.histogram(np.hstack(distances), bins=bins)
    hist_gas, _ = np.histogram(np.hstack(distances_gas), bins=bins)
    hist_gas[hist==0] = 1
    return hist / hist_gas


def get_gr_pbc(frames, bins, random_gas, box):
    bx, by, bz = box.ravel()
    neighbours = list(product([0, bx, -bx], [0, by, -by], [0, bz, -bz]))
    offset = 0
    distances = []
    distances_gas = []
    for frame in frames:
        if len(frame) > 2:
            dist_4d = [spatial.distance.cdist(frame, positions + n) for n in neighbours]
            dist = np.triu(np.min(dist_4d, axis=0), k=1)
            dist_gas = pdist(random_gas[offset : offset+len(frame)] + 1)
            offset += len(frame)
            distances.append(dist)
            distances_gas.append(dist_gas)
    hist, _ = np.histogram(np.hstack(distances), bins=bins)
    hist_gas, _ = np.histogram(np.hstack(distances_gas), bins=bins)
    hist_gas[hist==0] = 1
    return hist / hist_gas


def get_vanilla_gr(frames, tank, bins, random_size):
    """
    :param frames: positions of all particles in different frames, shape (frame, n, dim)
    :param tank: a static.Tank instance
    :param bins: the bins for the distance histogram
    :param random_size: the number of random gas particles
                        should be: len(frames) * particle_number_per_frame
    """
    random_gas = tank.random(random_size)
    return get_gr(frames, bins, random_gas)


def get_biased_gr(frames, positions, tank, bins, random_size, space_bin_number):
    """
    :param frames: positions of all particles in different frames, shape (frame, n, dim)
    :param positions: all positions in the entire movie, shape (N, 3)
    :param tank: a static.Tank instance
    :param bins: the bins for the distance histogram
    :param random_size: the number of random gas particles
                        should be: len(frames) * particle_number_per_frame
    """
    bins_xyz = (
        np.linspace(-tank.r_max, tank.r_max, space_bin_number+1, endpoint=True).ravel(),
        np.linspace(-tank.r_max, tank.r_max, space_bin_number+1, endpoint=True).ravel(),
        np.linspace(0, tank.z_max, space_bin_number+1, endpoint=True).ravel(),
    )
    random_gas = biased_discrete_nd(positions, bins_xyz, random_size)
    return get_gr(frames, bins, random_gas)


def get_mean_spd(velocity_frames, frame_number, min_number):
    """
    Calculate the average speed in one frame, given the velocities
    The calculation is done over many frames

    Args:
        velocity_frames (:obj:`numpy.ndarray`): velocity in different frames, shape (frame, n, dim)
        min_number (:obj:`numpy.ndarray`): only include frames if its particle number> min_number

    Return:
        :obj:`numpy.ndarray`: the average speed in each frame, shape (n, )
    """
    speeds = np.empty(frame_number)
    for i, velocity in enumerate(velocity_frames):
        if len(velocity) > min_number:
            speeds[i] = np.nanmean(np.linalg.norm(velocity, axis=1))
        else:
            speeds[i] = np.nan
    return np.nanmean(speeds)


def get_vicsek_order(velocity_frames, min_number):
    """
    Calculate the average Vicsek order parameters across different frames

    Args:
        velocity_frames (:obj:`list`): the velocity of particles at different frames
                                       "shape" (frame_num, particle_num, dimension)
                                       it is not a numpy array because `particle_num` in each frame is different
    """
    orders = np.empty(len(velocity_frames))
    for i, velocities in enumerate(velocity_frames):
        if len(velocities) > min_number:
            norms = np.linalg.norm(velocities, axis=1)
            norms[np.isclose(norms, 0)] = np.nan
            orientations = velocities / norms[:, np.newaxis]  # shape (n, dim)
            if False in np.logical_or(*np.isnan(orientations.T)):
                orders[i] = np.linalg.norm(np.nanmean(orientations, axis=0))  # shape (dim,)
            else:
                orders[i] = np.nan
        else:
            orders[i] = np.nan
    return orders


def fit_rot_acf(acf, delta):
    """
    Using linear fit to get the intersection of the acf function and x-axis

    Args:
        acf (:obj:`numpy.ndarray`): the acf function, shape (2, n) or (n, )
        delta (:obj:`float`): the range above/below 0, which will be fitted linearly

    Return:
        float: the relaxation time

    Example:
        >>> tau = np.arange(101)
        >>> acf = 1 - np.linspace(0, 2, 101)
        >>> data = np.array((tau, acf))
        >>> np.isclose(fit_rot_acf(acf, 0.1), 50)
        True
        >>> np.isclose(fit_rot_acf(data, 0.1), 50)
        True
    """
    if acf.ndim == 1:
        x = np.arange(len(acf))
        y = acf
    elif acf.ndim == 2:
        x, y = acf
    mask = np.zeros(y.shape, dtype=bool)
    mask[np.abs(y) < delta] = True
    turnning_point = np.argmax(np.diff(y) > 0)  # first element where slope > 0
    mask[turnning_point-1:] = False
    if len(x[mask]) == 0:
        return np.nan
    a, b = np.polyfit(x[mask], y[mask], deg=1)   # y = a * x + b
    return -b / a


def fit_acf_exp(data):
    """
    Use function :math:`y = exp(-x / a) \cdot b` to fit the acf data
    The fitting result a is a proxy to relaxation time

    Args:
        data (:obj:`numpy.ndarray`): the data to be fit, it can be either (tau, acf) or just acf
                           shape, (2, n) or (n,)

    Return:
        float: fitting parameter a, relaxation time

    Example:
        >>> tau = np.arange(100)
        >>> acf = np.exp(-tau / 10)
        >>> data = np.array((tau, acf))
        >>> np.isclose(fit_acf_exp(acf), 10)
        True
        >>> np.isclose(fit_acf_exp(data), 10)
        True
    """
    if data.ndim == 2:
        lag_time, acf = data
    elif data.ndim == 1:
        acf = data
        lag_time = np.arange(len(acf))
    popt, pcov = curve_fit(
        lambda x, a, b: np.exp(-x / a) * b,
        xdata = lag_time[1:],
        ydata = acf[1:],
        sigma = 1 / acf[1:]
    )
    return popt[0]
