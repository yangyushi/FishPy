#!/usr/bin/env python3
import pickle
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
    Calculate the best rotation to relate two sets of vectors

    See the paper [A solution for the best rotation to relate two sets of vectors] for detail

    all the points were treated equally, which means w_n = 0 (in the paper)

    Args:
        r1 (:obj:`numpy.ndarray`): a collection of 3D points, shape (n, 3)
        r2 (:obj:`numpy.ndarray`): a collection of 3D points, shape (n, 3)

    Return:
        :obj:`numpy.ndarray`: the best rotation matrix R, ``r1 @ R = r2``
    """
    R = r2.T @ r1
    u, a = np.linalg.eig(R.T @ R)  # a[:, i] corresponds to u[i]
    b = 1 / np.sqrt(u) * (R @ a)
    return (b @ a.T).T


def get_best_dilatation_rotation(r1, r2, init_guess=None):
    """
    Calculate the best dilation & rotation matrices between two sets of points

    .. code-block::

        (r1 @ Lambda) @ Rotation = r2

    Args:
        r1 (:obj:`numpy.ndarray`): a collection of 3D positions, shape (N, 3)
        r2 (:obj:`numpy.ndarray`): a collection of 3D positions, shape (N, 3)
        init_guess (:obj:`bool`): the initial guess for numerical optimisation

    Return:
        :obj:`tuple`: (dilation matrix Lambda, rotation matrix Rotation)
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
    maps = np.array(list(product(*bin_centres)))  # map from bin indices to values in the bin
    random_indices = ts.tower_sampling(size, discrete_pdf.astype(np.int64))
    random_numbers = maps[random_indices]
    return random_numbers + np.random.random(random_numbers.shape) * bin_width


def get_gr(frames, bins, random_gas):
    """
    Compare the radial distribution function from a collection of coordinates
        by comparing the distance distribution of particles in different frames with the random gas

    The density of the system and the random gas is chosen to be the same for each frame
        which reduced the effect of the possible tracking error

    Sample the random gas in clever ways to consider different boundaries

    Args:
        frames (:obj:`numpy.ndarray`): positions of particles in different frames, shape (frame, n, dim)
        bins (:obj:`numpy.ndarray` or int): the bins for the distance histogram
        random_gas (:obj:`numpy.ndarray`): positions of uncorrelated ideal gas, shape (N, 3)

    Return:
        :obj:`numpy.ndarray`: get the rdf values inside each bin
    """
    offset = 0
    distances = []
    distances_gas = []
    for frame in frames:
        if len(frame) > 2:
            dist = pdist(frame)
            dist_gas = pdist(random_gas[offset : offset+len(frame)])
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
    Args:
        frames (:obj:`numpy.ndarray`): positions of all particles in different frames, shape (frame, n, dim)
        tank (:obj:`Tank`): a static.Tank instance
        bins (:obj:`numpy.ndarray`): the bins for the distance histogram
        random_size (:obj:`int`): the number of random gas particles, its
                        should be: len(frames) * particle_number_per_frame
    """
    random_size = np.sum([len(f) for f in frames])
    random_gas = tank.random(random_size)
    return get_gr(frames, bins, random_gas)


def get_biased_gr(frames, positions, tank, bins, space_bin_number):
    """
    Args:
        frames (:obj:`numpy.ndarray`): positions of all particles in different frames, shape (frame, n, dim)
        positions (:obj:`numpy.ndarray`): all positions in the entire movie, shape (N, 3)
        tank (Tank): a static.Tank instance
        bins (:obj:`numpy.ndarray` or `int`): the bins for the distance histogram or the bin number
    """
    bins_xyz = (
        np.linspace(-tank.r_max, tank.r_max, space_bin_number+1, endpoint=True).ravel(),
        np.linspace(-tank.r_max, tank.r_max, space_bin_number+1, endpoint=True).ravel(),
        np.linspace(0, tank.z_max, space_bin_number+1, endpoint=True).ravel(),
    )
    frames = list(frames)
    random_size = np.sum([len(f) for f in frames])
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
        if len(velocity) > 0:
            spd = np.linalg.norm(velocity, axis=1)
            if np.logical_not(np.isnan(spd)).sum() > min_number:
                speeds[i] = np.nanmean(spd)
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


def pdist_pbc(positions, box):
    """
    Get the pair-wise distances of particles in a priodic boundary box

    Args:
        positiosn (:obj:`numpy.ndarray`): coordinate of particles, shape (N, dim)
        box (:obj:`float`): the length of the priodic bondary.
            The box should be cubic

    Return:
        :obj:`numpy.ndarray`: the pairwise distance, shape ( (N * N - N) / 2, ),
            use :obj:`scipy.spatial.distance.squareform` to recover the matrix form.
    """
    n, dim = positions.shape
    result = np.zeros(int((n * n - n) / 2), dtype=np.float64)
    for d in range(dim):
        dist_1d = pdist(positions[:, d][:, np.newaxis])
        dist_1d[dist_1d > box / 2] -= box
        result += np.power(dist_1d, 2)
    return np.sqrt(result)


class Trajectory():
    def __init__(self, time, positions, blur=None, velocities=None, blur_velocity=None):
        """
        Args:
            time (np.ndarray): frame number for each positon, dtype=int
            positions (np.ndarray): shape is (n_time, n_dimension)
            blur (float): applying gaussian_filter on each dimension along time axis
            velocities (np.ndarray): velocities at each time points
                                     this is ONLY possible for simulation data
        """
        if len(time) != len(positions):
            raise ValueError("Time points do not match the position number")
        self.time = time
        self.length = len(time)
        if blur:
            self.positions = ndimage.gaussian_filter1d(positions, blur, axis=0)
        else:
            self.positions = positions
        self.p_start = self.positions[0]
        self.p_end = self.positions[-1]
        self.t_start = self.time[0]
        self.t_end = self.time[-1]
        self.velocities = velocities
        if not isinstance(self.velocities, type(None)):
            self.v_end = self.velocities[-1]
        else:
            if blur_velocity:
                positions_smooth = ndimage.gaussian_filter1d(positions, blur_velocity, axis=0)
                self.v_end = (positions_smooth[-1] - positions_smooth[-2]) /\
                             (self.time[-1] - self.time[-2])
            else:
                self.v_end = (self.positions[-1] - self.positions[-2]) /\
                             (self.time[-1] - self.time[-2])

    def __len__(self):
        return len(self.positions)

    def __repr__(self):
        return f"trajectory@{id(self):x}"

    def __str__(self):
        return f"trajectory@{id(self):x}"

    def __add__(self, another_traj):
        """
        .. code-block::

            t1 + t2 == t2 + t1 == early + late
        """
        assert type(another_traj) == Trajectory, "Only Fish Trajectories can be added together"
        if self.t_start <= another_traj.t_end:  # self is earlier
            new_time = np.concatenate([self.time, another_traj.time])
            new_positions = np.concatenate([self.positions, another_traj.positions])
            return Trajectory(new_time, new_positions)
        elif self.t_end >= another_traj.t_start:  # self is later
            new_time = np.concatenate([another_traj.time, self.time])
            new_positions = np.concatenate([another_traj.positions, self.positions])
            return Trajectory(new_time, new_positions)
        else:  # there are overlap between time
            return self

    def predict(self, t):
        """
        predict the position of the particle at time t
        """
        assert t > self.t_end, "We predict the future, not the past"
        pos_predict = self.p_end + self.v_end * (t - self.t_end)
        return pos_predict

    def interpolate(self):
        if len(self.positions) == self.time[-1] - self.time[0]:
            return
        else:
            dimensions = range(self.positions.shape[1])
            pos_nd_interp = []
            for dim in dimensions:
                pos_1d = self.positions[:, dim]
                ti = np.arange(self.time[0], self.time[-1]+1, 1)
                pos_1d_interp = np.interp(x=ti, xp=self.time, fp=pos_1d)
                pos_nd_interp.append(pos_1d_interp)
            self.time = ti
            self.positions = np.vstack(pos_nd_interp).T

    def offset(self, shift):
        """
        offset all time points by an amount of shift
        """
        self.time += shift
        self.t_start += shift
        self.t_end += shift


class Movie:
    """
    Store both the trajectories and positions of experimental data

    .. code-block::

        Movie[f]             - the positions of all particles in frame f
        Movie.velocity(f)    - the velocities of all particles in frame f
        p0, p1 = Movie.indice_pair(f)
        Movie[f][p0] & Movie[f+1][p1] correspond to the same particles

    Attributes:
        trajs (:obj:`list` of :class:`Trajectory`)
        movie (:obj:`dict` of np.ndarray): hold the positions of particle in different frames
        __labels (:obj:`dict` of np.ndarray): hold the ID of particle in different frames
                                              label ``i`` corresponds to ``trajs[i]``
        __indice_pairs (:obj:`dict` of np.ndarray): the paired indices of frame ``i`` and frame ``i+1``

    Example:
        >>> def rand_traj(): return (np.arange(100), np.random.random((100, 3)))  # 3D, 100 frames
        >>> trajs = [rand_traj() for _ in range(5)]  # 5 random trajectories
        >>> movie = Movie(trajs)
        >>> movie[0].shape  # movie[f] = positions of frame f
        (5, 3)
        >>> movie.velocity(0).shape  # movie.velocity(f) = velocities of frame f
        (5, 3)
        >>> pairs = movie.indice_pair(0)  # movie.indice_pairs(f) = (labels in frame f, labels in frame f+1)
        >>> np.allclose(pairs[0], np.arange(5))
        True
        >>> np.allclose(pairs[1], np.arange(5))
        True
        >>> movie[0][pairs[0]].shape  # movie[f][pairs[0]] corresponds to movie[f+1][pairs[1]]
        (5, 3)
    """
    def __init__(self, trajs, blur=None, interpolate=False):
        self.trajs = self.__pre_process(trajs, blur, interpolate)
        self.__sniff()
        self.movie = {}
        self.__velocities = {}
        self.__labels = {}
        self.__indice_pairs = {}

    def __pre_process(self, trajs, blur, interpolate):
        new_trajs = []
        for t in trajs:
            if isinstance(t, Trajectory):
                if blur:
                    new_trajs.append(Trajectory(t.time, t.positions, blur=blur))
                else:
                    new_trajs.append(t)
            elif isinstance(t, dict):
                new_trajs.append(
                    Trajectory(t['time'], t['position'], blur=blur)
                )
            elif type(t) in (tuple, np.ndarray, list):
                new_trajs.append(Trajectory(t[0], t[1], blur=blur))
            else:
                raise TypeError("invalid type for trajectories")
        if interpolate:
            for traj in new_trajs:
                traj.interpolate()
        return new_trajs

    def __sniff(self):
        self.dim = self.trajs[0].positions.ndim
        self.max_frame = max([t.time.max() for t in self.trajs])
        self.size = len(self.trajs)

    def __len__(self): return self.max_frame

    def __process_velocities(self, frame):
        """
        Calculate *velocities* at different frames
        if particle ``i`` does not have a position in ``frame+1``, its velocity is ``np.nan``

        Args:
            frame (int): the specific frame to process

        Return:
            tuple: (velocity, (indices_0, indices_1))
                   velocity stores all velocity in ``frame``
        """
        if frame > self.max_frame - 1:
            raise IndexError("frame ", frame, "does not have velocities")
        else:
            position_0 = self[frame]
            position_1 = self[frame + 1]

            velocity = np.empty(position_0.shape)
            velocity.fill(np.nan)

            label_0 = self.__labels[frame]
            label_1 = self.__labels[frame + 1]
            label_intersection = [l for l in label_0 if l in label_1]

            if label_intersection:
                indices_0 = np.array([
                    np.where(li == label_0)[0][0] for li in label_intersection
                    ])
                indices_1 = np.array([
                    np.where(li == label_1)[0][0] for li in label_intersection
                    ])
                velocity[indices_0] = position_1[indices_1] - position_0[indices_0]
            else:
                indices_0 = np.empty(0)
                indices_1 = np.empty(0)

            return velocity, (indices_0, indices_1)

    def __get_positions_single(self, frame):
        if frame > self.max_frame:
            raise StopIteration
        elif frame in self.movie.keys():
            return self.movie[frame]
        else:
            positions = []
            labels = []
            for i, t in enumerate(self.trajs):
                if frame in t.time:
                    time_index = np.where(t.time == frame)[0][0]
                    positions.append(t.positions[time_index])
                    labels.append(i)
            if len(positions) == 0:
                positions = np.empty((0, self.dim))
            else:
                positions = np.array(positions)
            self.movie.update({frame: positions})

            labels = np.array(labels)
            self.__labels.update({frame: labels})
            return positions

    def __get_velocities_single(self, frame):
        if frame > self.max_frame - 1:
            raise IndexError(f"frame {frame} does not velocities")
        elif frame in self.__velocities.keys():
            return self.__velocities[frame]
        else:
            velocities, indice_pair = self.__process_velocities(frame)
            self.__velocities.update({frame: velocities})
            self.__indice_pairs.update({frame: indice_pair})
            return velocities

    def __get_slice(self, frame_slice, single_method):
        """
        Get the the slice equilivant of single_method
        """
        start = frame_slice.start if frame_slice.start else 0
        stop = frame_slice.stop if frame_slice.stop else self.max_frame + 1
        step = frame_slice.step if frame_slice.step else 1
        for frame in np.arange(start, stop, step):
            yield single_method(frame)

    def __getitem__(self, frame):
        if type(frame) in [int, np.int8, np.int16, np.int32, np.int64]:
            return self.__get_positions_single(frame)
        elif isinstance(frame, slice):
            return self.__get_slice(frame, self.__get_positions_single)
        else:
            raise KeyError(f"can't index/slice Movie with {type(frame)}")

    def velocity(self, frame):
        """
        Retireve velocity at given frame

        Args:
            frame (int / tuple): specifying a frame number or a range of frames

        Return:
            :obj:`list`: velocities of all particle in one frame or many frames,
                         the "shape" is (frame_num, particle_num, dimension)
                         it is not a numpy array because `particle_num` in each frame is different
        """
        if isinstance(frame, int):
            return self.__get_velocities_single(frame)
        elif isinstance(frame, tuple):
            if len(frame) in [2, 3]:
                frame_slice = slice(*frame)
                velocities = list(self.__get_slice(frame_slice, self.__get_velocities_single))
                return velocities
            else:
                raise IndexError(f"Invalid slice {frame}, use (start, stop) or (start, stop, step)")
        else:
            raise KeyError(f"can't index/slice Movie with {type(frame)}, use a Tuple")

    def label(self, frame):
        if frame > self.max_frame:
            return None
        elif frame in self.movie.keys():
            return self.__labels[frame]
        else:
            self[frame]
            return self.__labels[frame]

    def indice_pair(self, frame):
        """
        Return two set of indices, idx_0 & idx_1
        ``Movie[frame][idx_0]`` corresponds to ``Movie[frame + 1][idx_1]``

        Args:
            frame (int): the frame number

        Return:
            :obj:`tuple` of np.ndarray: the indices in ``frame`` and ``frame + 1``
        """
        if frame > self.max_frame - 1:
            raise IndexError(f"frame {frame} does not have a indices pair")
        elif frame in self.__indice_pairs.keys():
            return self.__indice_pairs[frame]
        else:
            velocities, indice_pair = self.__process_velocities(frame)
            self.__velocities.update({frame: velocities})
            self.__indice_pairs.update({frame: indice_pair})
            return indice_pair

    def make(self):
        """
        Go through all frames, making code faster with the object
        """
        for i in range(len(self)):
            self[i]

    def load(self, filename: str):
        """
        Load a saved file in the hard drive
        """
        with open(filename, 'rb') as f:
            movie = pickle.load(f)
        self.trajs = movie.trajs
        self.movie = movie.movie
        self.__velocities = movie.__velocities
        self.__labels = movie.__labels
        self.__indice_pairs = movie.__indice_pairs
        self.__sniff()

    def save(self, filename: str):
        """
        Save all data using picle
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def update_trajs(self, blur=0, interpolate=False):
        """
        Reconstruct ``self.trajs``, typically used if :class:`Trajectory` is modified
        """
        new_trajs = []
        for t in self.trajs:
            new_trajs.append(Trajectory(t.time, t.positions, blur=blur))
        if interpolate:
            for traj in new_trajs:
                traj.interpolate()
        self.trajs = new_trajs


class SimMovie:
    def __init__(self, positions, velocities):
        """
        Args:
            positions(np.ndarray): positions of all particles in all frames, shape (frame, number, dimension)
            velocities(np.ndarray): velocities of all particles in all frames, shape (frame, number, dimension)
        """
        self.positions = positions
        self.velocities = velocities
        time = np.arange(positions.shape[0])
        self.trajs = [Trajectory(time, pos, velocities=vol) for pos, vol in zip(
            np.transpose(positions, (1, 0, 2)),
            np.transpose(velocities, (1, 0, 2)),
        )]
        self.__labels = np.arange(self.positions.shape[1])
        self.max_frame = self.positions.shape[0]

    def __len__(self): return self.positions.shape[0]

    def __getitem__(self, frame): return self.positions[frame]

    def velocity(self, frame):
        if isinstance(frame, tuple):
            return self.velocities[slice(*frame)]
        else:
            return self.velocities[frame]

    def label(self, frame): return self.__labels

    def indice_pair(self, frame): return (self.__labels, self.__labels)

    def make(self): pass

    def load(self, filename):
        with open(filename, 'rb') as f:
            movie = pickle.load(f)
        self.positions = movie.positions
        self.velocities = movie.velocities

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def export_xyz(self, filename, comment='none'):
        f = open(filename, 'w')
        for frame in self.positions:
            x, y, z = frame.T
            f.write('{}\n'.format(len(x)))
            f.write('{}\n'.format(comment))
            for i in range(len(x)):
                f.write('A\t{:.8e}\t{:.8e}\t{:.8e}\n'.format(x[i], y[i], z[i]))
        f.close()
