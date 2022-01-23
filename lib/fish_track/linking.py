#!/usr/bin/env python3
import os
import pickle
import numpy as np
import trackpy as tp
from numba import njit
from typing import List
from scipy import ndimage
from scipy.sparse import coo_matrix
from joblib import delayed, Parallel
from numba.typed import List as nList
from scipy.spatial.distance import cdist
from .nrook import solve_nrook, solve_nrook_dense
import matplotlib.pyplot as plt
try:
    from fish_corr import Movie, SimMovie
except ImportError:
    pass


@njit
def get_trajectory(labels, frames, target: int):
    time, positions = nList(), nList()
    for t in range(len(labels)):
        index = -1
        label = labels[t]
        frame = frames[t]
        index, found = -1, False

        for i, l in enumerate(label):
            if l == target:
                index, found = i, True
                break

        if index >= 0:
            positions.append(frame[index])  # (xy, xy2, ..., xyn)
            time.append(t)
        elif found:
            return time, positions
    return time, positions


class Trajectory():
    """
    Bundle handy methods with trajectory data
    """
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
        if not isinstance(self.velocities, type(None)):  # for simulation data
            self.v_end = self.velocities[-1]
            self.v_start = self.velocities[0]
        else:
            if blur_velocity:
                positions_smooth = ndimage.gaussian_filter1d(positions, blur_velocity, axis=0)
                self.v_end = (positions_smooth[-1] - positions_smooth[-2]) /\
                             (self.time[-1] - self.time[-2])
                self.v_start = (positions_smooth[1] - positions_smooth[0]) /\
                             (self.time[1] - self.time[0])
            else:
                self.v_end = (self.positions[-1] - self.positions[-2]) /\
                             (self.time[-1] - self.time[-2])
                self.v_start = (self.positions[1] - self.positions[0]) /\
                             (self.time[1] - self.time[0])

    def predict(self, t):
        """
        predict the position of the particle at future time-point t
        """
        assert t > self.t_end, "We predict the future, not the past"
        pos_predict = self.p_end + self.v_end * (t - self.t_end)
        return pos_predict

    def retrace(self, t):
        """
        retrace the position of the particle at previous time-point t
        """
        assert t < self.t_start, "We retrace the past, not the future"
        pos_retrace = self.p_start + self.v_start * (t - self.t_start)
        return pos_retrace

    def distance(self, traj):
        """
        Calculate the distance between self and another trajectory

        the distance is defined as the distance where the earlier trajectory
            travel to future, and the later trajectory travel to past, and the
            distance where they meet in time.

        The purpose of such calculation is to improve the linking success rate.
        """
        if self.t_end <= traj.t_start:  # self is earlier
            t = (traj.t_start - self.t_end) / 2
            p0 = self.p_end + self.v_end * t  # forwards
            p1 = traj.p_start - traj.v_start * t  # backward
            return np.linalg.norm(p0 - p1)
        elif self.t_start >= traj.t_end:  # self is later
            t = (self.t_start - traj.t_end) / 2
            p0 = traj.p_end + traj.v_end * t  # forwards
            p1 = self.p_start - self.v_start * t  # backward
            return np.linalg.norm(p0 - p1)
        else:
            return np.inf

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

    def offset(self, time):
        """
        offset all time
        """
        self.time += time
        self.t_start += time
        self.t_end += time
        return self

    def interpolate(self):
        if len(np.unique(np.diff(self.time))) == 0:
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

    def break_into_two(self, time_index):
        """
        Break one trajectory into two trajectories.

        Args:
            time_index (int): the *index* of self.time at which the trajectory
                will be break down.

        Return:
            list: a list of two trajecotires
        """
        p1 = self.positions[:time_index]
        t1 = self.time[:time_index]
        p2 = self.positions[time_index:]
        t2 = self.time[time_index:]
        if (len(t1) > 2) and (len(t2) > 2):
            return [Trajectory(t1, p1), Trajectory(t2, p2)]
        else:
            return [self]  # do not break if trying to break tail


class ActiveLinker():
    """
    Link positions into trajectories following 10.1007/s00348-005-0068-7
    Works with n-dimensional data in Euclidean space
    """
    def __init__(self, search_range):
        self.search_range = search_range
        self.labels = None
        self.trajectories = None

    def link(self, frames):
        """
        Getting trajectories from positions in different frames

        Args:
            frames (:obj:`list` of :obj:`numpy.ndarray`): the positions of particles in different frames from the experimental data

        Return:
            :obj:`list`: a collection of trajectories. Each trajectory is represented by a list, [time_points, positions]
        """
        self.labels = self.__get_labels(frames)
        self.trajectories = self.__get_trajectories(self.labels, frames)
        return self.trajectories

    def __predict(self, x1, x0=None, xp=None):
        """
        predict position in frame 2 (x2)
        according to x1, x0, and xp
        """
        if isinstance(x0, type(None)):
            x0 = x1
        if isinstance(xp, type(None)):
            xp = 2 * x0 - x1
        return 3 * x1 - 3 * x0 + xp

    def __get_link_f3(self, f0, f1, f2, links=None):
        if len(f2) == 0:
            return []
        if isinstance(links, type(None)):
            links = []
        new_links = []
        for i0, p0 in enumerate(f0):
            if i0 in [l[0] for l in links]:
                continue
            dist_1 = cdist([self.__predict(p0)], f1)[0]
            candidates_1 = f1[dist_1 < self.search_range]
            labels_1 = np.arange(len(f1))
            labels_1 = labels_1[dist_1 < self.search_range]
            costs = np.empty(labels_1.shape)
            for i, (l1, p1) in enumerate(zip(labels_1, candidates_1)):
                if l1 not in [l[1] for l in links + new_links]:
                    dist_2 = cdist([self.__predict(p1, p0)], f2)[0]
                    costs[i] = np.min(dist_2)
                else:
                    costs[labels_1==l1] = np.inf
            if len(costs) > 0:
                if np.min(costs) < self.search_range * 2:
                    i1 = labels_1[np.argmin(costs)]
                    new_links.append((i0, i1))
        return new_links

    def __get_link_f4(self, fp, f0, f1, f2, old_links):
        if len(f2) == 0:
            return []
        new_links = []
        for ip, i0 in old_links:
            p0 = f0[i0]
            dist_1 = np.linalg.norm(
                    self.__predict(p0, fp[ip])[np.newaxis, :] - f1,  # (n, 3)
                    axis=1
                    )
            candidates_1 = f1[dist_1 < self.search_range]
            if len(candidates_1) == 0:
                continue
            labels_1 = np.arange(len(f1))[dist_1 < self.search_range]
            costs = np.empty(labels_1.shape)
            for i, (l1, p1) in enumerate(zip(labels_1, candidates_1)):
                if l1 not in [l[1] for l in new_links]:  # if l1, p1 is not conflict with other links
                    dist_2 = np.linalg.norm(
                            self.__predict(p1, p0, fp[ip])[np.newaxis, :] - f2,
                            axis=1
                            )
                    costs[i] = np.min(dist_2)
                else:
                    costs[labels_1==l1] = np.inf
            if (len(costs) > 0) and (np.min(costs) < self.search_range * 2):
                i1 = labels_1[np.argmin(costs)]
                new_links.append((i0, i1))
        new_links += self.__get_link_f3(f0, f1, f2, new_links)  # get links start in frame 0
        return new_links

    def __get_links(self, fp, f0, f1, f2, links):
        """
        Get links in two successive frames

        Args:
            fp (:obj:`numpy.ndarray`): previous frame, (dim, n)
            f0 (:obj:`numpy.ndarray`): current frame, (dim, n)
            f1 (:obj:`numpy.ndarray`): next frame, (dim, n)
            f2 (:obj:`numpy.ndarray`): second next frame, (dim, n)
            links (list): index correspondence for particles between fp to f0

        Return:
            list: link from f0 to f1
        """
        if isinstance(fp, type(None)):
            return self.__get_link_f3(f0, f1, f2)
        else:
            return self.__get_link_f4(fp, f0, f1, f2, links)

    def __get_all_links(self, frames):
        """
        Get all possible links between all two successive frames
        """
        frame_num = len(frames)
        links = None
        for n in range(frame_num - 2):
            if n == 0:
                # fp = previous frame
                fp, f0, f1, f2 = None, frames[n], frames[n+1], frames[n+2]
            else:
                fp, f0, f1, f2 = f0, f1, f2, frames[n+2]
            links = self.__get_links(fp, f0, f1, f2, links)
            yield links

    def __get_labels(self, frames):
        links_all = self.__get_all_links(frames)
        labels = [ np.arange(len(frames[0])) ]  # default label in first frame
        label_set = set(labels[0].tolist())
        if len(label_set) > 0:
            new_label = max(label_set) + 1
        else:
            new_label = 0
        for frame, links in zip(frames[1:-1], links_all):
            old_labels = labels[-1]
            new_labels = np.empty(len(frame), dtype=int) # every particle will be labelled
            slots = np.arange(len(frame))
            linked = [l[1] for l in links]
            linked.sort()
            for l in links:
                new_labels[l[1]] = old_labels[l[0]]
            for s in slots:
                if s not in linked:
                    new_labels[s] = new_label
                    label_set.add(new_label)
                    new_label += 1
            labels.append(new_labels)
        return labels

    def __get_trajectories_slow(self, labels, frames):
        frame_nums = len(labels)
        max_value = np.hstack(labels).max()
        trajectories = []
        for i in range(max_value):  # find the trajectory for every label
            traj = {'time': None, 'position': None}
            time, positions = [], []
            for t, (label, frame) in enumerate(zip(labels, frames)):
                if i in label:
                    index = label.tolist().index(i)
                    positions.append(frame[index])  # (xy, xy2, ..., xyn)
                    time.append(t)
            traj['time'] = np.array(time)
            traj['position'] = np.array(positions)
            trajectories.append(traj)
        return trajectories

    @staticmethod
    def __get_trajectories(labels, frames):
        """
        This is used in ActiveLinker

        .. code-block::
            trajectory = [time_points, positions]
            plot3d(*positions, time_points) --> show the trajectory in 3D

        shape of centres: (N x dim)
        """
        max_value = np.hstack(labels).max()
        labels_numba = nList()
        frames_numba = nList()

        for label, frame in zip(labels, frames):
            labels_numba.append(label.copy())
            frames_numba.append(frame.copy())

        trajectories = []
        for target in range(max_value + 1):  # find the trajectory for every label
            result = get_trajectory(labels_numba, frames_numba, target)
            time, positions = result
            trajectories.append(
                (np.array(time), np.array(positions))
            )
        return trajectories


class TrackpyLinker():
    """
    Linking positions into trajectories using Trackpy
    Works with 2D and 3D data. High dimensional data not tested.
    (no expiment available)
    """
    def __init__(self, max_movement, memory=0, max_subnet_size=30, **kwargs):
        self.max_movement = max_movement
        self.memory = memory
        self.max_subnet_size = max_subnet_size
        self.kwargs = kwargs

    @staticmethod
    def _check_input(positions, time_points, labels):
        """
        Make sure the input is proper and sequence in time_points are ordered
        """
        assert len(positions) == len(time_points), "Lengths are not consistent"
        if not isinstance(labels, type(None)):
            assert len(positions) == len(labels), "Lengths are not consistent"
            for p, l in zip(positions, labels):
                assert len(p) == len(l), "Labels and positions are not matched"
        time_points = np.array(time_points)
        order_indice = time_points.argsort()
        ordered_time = time_points[order_indice]
        positions = list(positions)
        positions.sort(key=lambda x: order_indice.tolist())
        return positions, ordered_time, labels

    def __get_trajectory(self, value, link_result, positions, time_points, labels):
        with_label = False
        if isinstance(labels, type(None)):
            traj = [[], []]
        else:
            traj = [[], [], []]
            with_label = True
        for frame in link_result:
            frame_index, link_labels = frame
            if value in link_labels:
                number_index = link_labels.index(value)
                traj[0].append(time_points[frame_index])
                traj[1].append(positions[frame_index][number_index])
                if with_label:
                    current_label = labels[frame_index][link_labels.index(value)]
                    traj[2].append(current_label)
        traj[0] = np.array(traj[0])
        traj[1] = np.array(traj[1])
        return traj

    def __get_trajectories(self, link_result, positions, time_points, labels):
        trajectories = []
        total_labels = []
        for frame in link_result:
            frame_index, link_labels = frame
            total_labels.append(link_labels)
        for value in set(np.hstack(total_labels)):
            traj = self.__get_trajectory(value, link_result, positions, time_points, labels)
            trajectories.append(traj)
        return trajectories

    def link(self, positions, time_points=None, labels=None):
        """
        Args:
            positions (np.ndarray): shape (time, num, dim)
            time_points (np.ndarray): shape (time, ), time_points may not be continues
            labels (np.ndarray): if given, the result will have a 'label' attribute
                                 which specifies the label values in different frames
                                 [(frame_index, [labels, ... ]), ...], shape (time, num)
        """
        if isinstance(time_points, type(None)):
            time_points = np.arange(len(positions))
        pos, time, labels = self._check_input(positions, time_points, labels)
        tp.linking.Linker.MAX_SUB_NET_SIZE = self.max_subnet_size
        link_result = tp.link_iter(pos, search_range=self.max_movement, memory=self.memory, **self.kwargs)
        return self.__get_trajectories(list(link_result), pos, time, labels)


def sort_trajectories(trajectories):
    """
    Sort trajectories according to the first time point in each traj

    Args:
        trajectories (List): a collection of :class:`Trajectory` or obj:`tuple`
            which contains the time points and positions of each trajectory

    Return:
        list: the sorted trajectories, each trajectory maintains the same type.
    """
    start_time_points = np.empty(len(trajectories))
    if type(trajectories[0]) == Trajectory:
        for i, traj in enumerate(trajectories):
            start_time_points[i] = traj.t_start
    elif type(trajectories[0]) == tuple:
        for i, traj in enumerate(trajectories):
            start_time_points[i] = traj[0][0]
    else:
        raise TypeError("Unsupported trajectory type:", type(trajectories[0]))
    sorted_indices = np.argsort(start_time_points)
    return [trajectories[si] for si in sorted_indices]


def squeeze_sparse(array):
    """
    Given the indices in a row or column from a sparse matrix
    Remove the blank rows/columns

    .. code-block::

        array[i + 1] >= array[i]

    Args:
        array (:obj:`numpy.ndarray`): the indices of columes/rows in a sparse matrix

    Return:
        :obj:`numpy.ndarray`: the matrix without blank columes/rows

    Example:
        >>> input = np.array([44, 53, 278, 904, 1060, 2731])
        >>> output = np.array([0, 1, 2, 3, 4, 5])
        >>> np.allclose(output, squeeze_sparse(output))
        True
    """
    ## 1, 3, 2, 0 --argsort-> 3, 0, 2, 1 --argsort-> 1, 3, 2, 0
    sort_indices = np.argsort(array)
    remap = np.argsort(sort_indices)
    array_sorted = array[sort_indices]

    result = np.empty(array.shape, dtype=int)
    result[0] = 0
    for i in range(1, len(array)):
        if array_sorted[i] == array_sorted[i - 1]:
            result[i] = result[i-1]
        elif array_sorted[i] > array_sorted[i - 1]:
            result[i] = result[i-1] + 1
        else:
            raise RuntimeError("array[i + 1] < array[i]")

    return result[remap]


def cost_dist(traj_1, traj_2):
    """
    Args:
        traj_1 (Trajectory): earlier trajectory
        traj_2 (Trajectory): later trajectory

    Return:
        float: the cost value
    """
    d1 = np.linalg.norm(traj_1.predict(traj_2.t_start) - traj_2.p_start)
    d2 = np.linalg.norm(traj_2.retrace(traj_1.t_end) - traj_1.p_end)
    d3 = traj_1.distance(traj_2)
    return min((d1, d2, d3))


def cost_xu(traj_1, traj_2):
    shift_x = traj_1.predict(traj_2.t_start) - traj_1.p_start
    shift_v = (traj_1.v_end - traj_2.v_start) * (traj_2.t_start - traj_1.t_end)
    return np.sqrt(np.sum(shift_x ** 2 + shift_v ** 2))

def time_filter_link(traj_1, traj_2, dt):
    """
    Return ture if two trajectories would form portational link
    """
    gap = traj_2.t_start - traj_1.t_end
    return (gap > 0) and (gap <= dt)


def time_filter_merge(traj_1, traj_2, dt):
    """
    Return ture if two trajectories would form portational link
    """
    gap =traj_1.t_end - traj_2.t_start
    return (gap >=0) and (gap <= dt)


def build_dist_matrix_sparse(
        trajs_sorted, dt, dx, cost, time_filter, debug=False
):
    """
    Build a sparse distance matrix, then squeeze it

    Args:
        trajs_sorted (list): trajectories obtained from :meth:`sort_trajectories`
        dt (int): if the time[-1] of traj #1 + dt > time[0] of traj #2
                  consider a link being possible
        dx (float): if within dt, the distance between
                    * trajectory #1's prediction and
                    * trajectory #2's first point
                    is smaller than dx, assign a link
        cost (callable): the function to calculate cost between two
            trajectories, trajectories with smaller cost will be linked

    Return:
        tuple: (distance matrix (scipy.sparse.coo_matrix), row_map, col_map)
               The ``row_map`` and ``col_map`` are used to map
               from squeezed distance matrix to origional distance matrix
    """
    values, rows, cols = [], [], []
    for i, traj_1 in enumerate(trajs_sorted):
        for j, traj_2 in enumerate(trajs_sorted[i+1:]):
            if time_filter(traj_1, traj_2, dt):
                distance = cost(traj_1, traj_2)
                if distance < dx:
                    rows.append(i)
                    cols.append(i+j+1)
                    values.append(distance)
    if len(values) == 0:
        return np.empty(0), {}, {}, np.empty((0, 2), dtype=int)
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    values = np.array(values, dtype=float)
    rows, cols, values, unique_links = solve_unique(
        rows, cols, values, debug=debug
    )
    if len(rows) == 0:
        return np.empty((0, 0)), {}, {}, unique_links
    rows_sqz = squeeze_sparse(rows)
    cols_sqz = squeeze_sparse(cols)
    # map from squeezed to origional
    row_map = {r1 : r0 for r0, r1 in zip(rows, rows_sqz)}
    col_map = {c1 : c0 for c0, c1 in zip(cols, cols_sqz)}
    larger = max(max(rows_sqz), max(cols_sqz)) + 1
    dist_mat = coo_matrix(
        (values, (rows_sqz, cols_sqz)), shape=(larger, larger), dtype=float
    ).toarray()
    return dist_mat, row_map, col_map, unique_links


def reduce_network(network):
    """
    Network is compose of linkes that looks like,

    .. code-block:: py

        [(i_1, j_1), (i_2, j_2) ...]

    The network is **sorted** with

    .. code-block:: py

        i_n > ... > i_2 > i_1

    Args:
        network (:obj:`numpy.ndarray`): a collection of links, shape (n, 2)

    Return:
        :obj:`list` of :obj:`numpy.ndarray`: the reduced network

    Example:
        >>> input = np.array([(0, 1), (1, 3), (3, 5), (4, 5), (6, 7), (7, 8)])
        >>> output = np.array([(0, 1, 3, 5), (4, 5), (6, 7, 8)])
        >>> [np.allclose(i, o) for i, o in zip(reduce_network(input), output)]
        [True, True, True]
    """
    reduced, to_skip = [], []
    for i, link in enumerate(network):
        if i in to_skip:
            continue
        new_link = link.copy()
        ll = np.where(link[-1] == network[:, 0])[0]  # link of link
        while len(ll) > 0:
            to_skip.append(ll)
            new_link = np.hstack((new_link, network[ll[0], 1]))
            ll = np.where(new_link[-1] == network[:, 0])[0]
        reduced.append(new_link)
    return reduced


def choose_network(distance_matrix, networks) -> np.ndarray:
    """
    Choose the network with minimum distance amonest a collection of networks

    Args:
        distance_matrix(:obj:`numpy.ndarray`): the distances for different links.
            distance_matrix[link] --> distance of this link
        networks (:obj:`numpy.ndarray`): a collection of networks, a network
            is a collection of links. The shape is (n_network, n_link, 2)

    Return:
        :obj:`numpy.ndarray`: the network with minimum distance, shape (n_link, 2)
    """
    distances = []
    for network in networks:
        costs = distance_matrix[tuple(network.T)]
        dist_total = distance_matrix[network].sum()
        distances.append(dist_total)
    return networks[np.argmin(distances)]


def apply_network_link(trajectories, network):
    """
    Args:
        trajectories (:obj:`list` of :obj:`Trajectory`): unlinked trajectories
        network (:obj:`list` of :obj:`numpy.ndarray`): the chosen network to link trajectories
            Each network may have different sizes

    Return:
        :obj:`list` of :obj:`Trajectory`: trajectories that were connected according
        to the given network
    """
    new_trajs = []
    to_modify = np.hstack(network)
    for link in network:
        to_sum = [trajectories[idx] for idx in link]
        for t in to_sum[1:]:
            to_sum[0] = to_sum[0] + t
        new_trajs.append(to_sum[0])
    for i, t in enumerate(trajectories):
        if i not in to_modify:
            new_trajs.append(t)
    return new_trajs


def apply_network_merge(trajectories, network):
    """
    Merge the trajectories. The earlier trajecotires will be clipped and
        connectd to later trajectories

    Args:
        trajectories (:obj:`list` of :obj:`Trajectory`): unmerged trajectories
        network (:obj:`list` of :obj:`numpy.ndarray`): the chosen network to
            merge trajectories. Each network may have different sizes.

    Return:
        :obj:`list` of :obj:`Trajectory`: trajectories that were merged
            according to the given network
    """
    new_trajs = []
    to_modify = np.hstack(network)
    for link in network:
        to_sum = [trajectories[idx] for idx in link]
        for i, traj_1 in enumerate(to_sum[:-1]):
            traj_2 = to_sum[i + 1]
            t1 = np.argmin(np.abs(traj_1.time - traj_2.t_start))
            to_sum[i] = Trajectory(traj_1.time[:t1], traj_1.positions[:t1])
        for t in to_sum[1:]:
            to_sum[0] = to_sum[0] + t
        new_trajs.append(to_sum[0])
    for i, t in enumerate(trajectories):
        if i not in to_modify:
            new_trajs.append(t)
    return new_trajs


def solve_unique(rows, cols, values, debug=False):
    """
    Find the link between two trajecotires that are both one-to-one and onto

    That is to say, trajectory A only link to B, and B is only only linked by A.

    Example:

    ..code-block:: python

          A           B
       ------>     ------->
       (first)     (second)


    Args:
        rows (np.ndarray): The indices of the first trajectories
        cols (np.ndarray): The indices of the second trajectories
        values (np.ndarray): The distances between the prediction of
            first trajectory and the start of the second trajectory.
            Essentially this is the error.

    Return:
        tupel: (
            not-unique row indices, not-unique col indices,
            not-unique distances, unique_links
        )

    """
    unsolved_indices, unique_links = [], []
    elements_row, indices_row, counts_row = np.unique(rows, return_index=True, return_counts=True)
    elements_col, indices_col, counts_col = np.unique(cols, return_index=True, return_counts=True)
    indices_row_unique = indices_row[counts_row==1]
    indices_col_unique = indices_col[counts_col==1]
    unique_indices = np.intersect1d(indices_row_unique, indices_col_unique, assume_unique=True)
    unsolved_indices = np.setdiff1d(np.arange(len(rows)), unique_indices)
    if debug:
        print(f"{len(unique_indices)} unique indices out of {len(rows)} pairs")
    if len(unique_indices) == 0:
        unique_links = np.empty((0, 2), dtype=int)
    else:
        unique_links = np.array([
            (rows[i], cols[i]) for i in unique_indices
        ], dtype=int)
    return rows[unsolved_indices], cols[unsolved_indices], values[unsolved_indices], unique_links


def relink(trajectories, dx, dt, blur=None, blur_velocity=None, cost='dist', debug=False):
    """
    Re-link short trajectories into longer ones.

    Args:
        trajectories (:obj:`list`): A collection of trajectories, where
            each trajectory is stored in a tuple in the form of, (time, positions)
        dx (:obj:`float`): distance threshold, the only trajectories whose 'head' and 'tail'
            is smaller than dx in space were considered as a possible link
        dt (:obj:`int`): time threshold, the only trajectories whose 'head' and 'tail'
            is smaller than dt in time were considered as a possible link
        blur (:obj:`float` or :obj:`bool`): if blur is provided, all trajectories were filtered using
            a gaussian kernel, whose sigma is the value of ``blur``

    Return:
        :obj:`list` of :obj:`tuple`: The relink trajectories.
            Each trajectory is stored in a tuple, (time, positions)
    """

    if type(trajectories[0]) in (tuple, np.ndarray, list):
        trajs = [ Trajectory( t[0], t[1], blur=blur, blur_velocity=blur_velocity) for t in trajectories if len(t[0]) > 1 ]
    else:
        raise TypeError("Invalid Trajectory Data Type")

    trajs_ordered = sort_trajectories(trajs)

    if cost == 'xu':
        cost = cost_xu
    elif cost == 'dist':
        cost = cost_dist
    else:
        raise ValueError("Unsupported cost option:", cost)

    dist_mat, row_map, col_map, unique_links = build_dist_matrix_sparse(
        trajs_ordered, dx=dx, dt=dt, debug=debug,
        cost=cost, time_filter=time_filter_link
    )

    if (len(dist_mat) == 0) and (len(unique_links) == 0):
        if debug: print("No conflict")
        return [(t.time, t.positions) for t in trajs_ordered]

    elif (len(dist_mat) == 0):  # all the links were unique
        if debug: print("all links are unique")
        network = unique_links
        reorder_indices = np.argsort(network[:, 0])  # sort for `reduce_network`
        reduced = reduce_network(network[reorder_indices])
        new_trajs = apply_network_link(trajs_ordered, reduced)
        return [(t.time, t.positions) for t in new_trajs]

    else:  # conflict links exist
        if debug: print("solving conflicts")
        if (len(dist_mat) >= 100) and debug:
            print("Solving LARGE linking matrix")

        max_row = max(row_map.keys()) + 1
        networks = solve_nrook_dense(dist_mat.astype(bool), max_row=max_row)

        best = choose_network(dist_mat, networks)

        if debug:
            print("best shape (should be n, 2)", best.shape)
            print("unique_links shape (should be n, 2)", unique_links.shape)

        for i, pair in enumerate(best):
            best[i, 0] = row_map[pair[0]]
            best[i, 1] = col_map[pair[1]]

        network = np.concatenate((unique_links, best))
        reorder_indices = np.argsort(network[:, 0])  # sort for `reduce_network`

        reduced = reduce_network(network[reorder_indices])
        new_trajs = apply_network_link(trajs_ordered, reduced)

        return [(t.time, t.positions) for t in new_trajs]


def cost_merge(traj_1, traj_2):
    t1 = np.argmin(np.abs(traj_1.time - traj_2.t_start))
    p1 = traj_1.positions[t1]
    return np.linalg.norm(p1 - traj_2.p_start)


def merge_trajectories(trajectories, dx, dt, debug=False):
    """
    Merge two trajectories, t1 and t2, if the end of t1 matches the head of t2.

    Args:
        trajectories (:obj:`list`): A collection of trajectories.
            Each trajectory is stored in a tuple, (time, positions)
        dx (:obj:`float`): distance threshold, t1 and t2 were considered to be
            matched if the distance between the end of t1 and the head of t2 were
            is smaller than dx.

    Return:
        (:obj:`list`): a collection of merged trajectories
    """
    if type(trajectories[0]) in (tuple, np.ndarray, list):
        trajs = [Trajectory(t[0], t[1]) for t in trajectories if len(t[0]) > 1 ]
    else:
        raise TypeError("Invalid Trajectory Data Type")

    trajs_ordered = sort_trajectories(trajs)

    dist_mat, row_map, col_map, unique_links = build_dist_matrix_sparse(
        trajs_ordered, dx=dx, dt=dt, debug=debug,
        cost=cost_merge, time_filter=time_filter_merge
    )

    if (len(dist_mat) == 0) and (len(unique_links) == 0):
        if debug: print("Nothing to merge")
        return [(t.time, t.positions) for t in trajs_ordered]

    elif (len(dist_mat) == 0):  # all the links were unique
        if debug: print("All links are unique")
        network = unique_links
        reorder_indices = np.argsort(network[:, 0])  # sort for `reduce_network`
        reduced = reduce_network(network[reorder_indices])
        new_trajs = apply_network_merge(trajs_ordered, reduced)
        return [(t.time, t.positions) for t in new_trajs]

    else:  # conflict links exist
        if debug: print("solving conflicts")
        if (len(dist_mat) >= 100) and debug:
            print("Solving LARGE linking matrix")

        max_row = max(row_map.keys()) + 1
        networks = solve_nrook_dense(dist_mat.astype(bool), max_row=max_row)

        best = choose_network(dist_mat, networks)

        if debug:
            print("best shape (should be n, 2)", best.shape)
            print("unique_links shape (should be n, 2)", unique_links.shape)

        for i, pair in enumerate(best):
            best[i, 0] = row_map[pair[0]]
            best[i, 1] = col_map[pair[1]]

        network = np.concatenate((unique_links, best))
        reorder_indices = np.argsort(network[:, 0])  # sort for `reduce_network`

        reduced = reduce_network(network[reorder_indices])
        new_trajs = apply_network_merge(trajs_ordered, reduced)

        return [(t.time, t.positions) for t in new_trajs]


def segment_trajectories(trajectories, window_size, max_frame):
    """
    Split trajectories into different segments according to their starting time points

    If there are 5400 frames, with a `window_size = 500`, then the edges of each segment are

    .. code-block:: py

        [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5400]

    Args:
        trajectories (:obj:`list`): A collection of trajectories.
            Each trajectory is stored in a tuple, (time, positions)
        window_size (:obj:`int`): the frame number size of each segment for
            the relink of the 1st iteration
        max_frame (:obj:`int`): the maximum number of frames

    Return:
        :obj:`list`: a list of trajectories in different segments
    """
    edges = [window_size * i for i in range(max_frame // window_size + 1)]
    if max_frame % window_size:
        edges.append(max_frame)
    traj_segments = [[] for _ in range(len(edges) - 1)]
    segment_indices = np.digitize(
        [t[0][0] for t in trajectories], bins=edges, right=False
    ) - 1
    for traj, si in zip(trajectories, segment_indices):
        traj_segments[si].append(traj)
    traj_segments = [ts for ts in traj_segments if len(ts) > 0]
    return traj_segments


def relink_by_segments(trajectories, window_size, max_frame, dx, dt,
                       blur=None, blur_velocity=None, cost='dist', debug=False):
    """
    Re-link short trajectories into longer ones.

    The task is firstly separated into different segments with equal size, then combined

    Args:
        trajectories (:obj:`list`): A collection of trajectories.
            Each trajectory is stored in a tuple, (time, positions)
        window_size (:obj:`int`): the frame number size of each segment for
            the relink of the 1st iteration
        max_frame (:obj:`int`): the maximum number of frames
        dx (:obj:`float`): distance threshold, the only trajectories whose 'head' and 'tail'
            is smaller than dx in space were considered as a possible link
        dt (:obj:`int`): time threshold, the only trajectories whose 'head' and 'tail'
            is smaller than dt in time were considered as a possible link
        blur (:obj:`float` or :obj:`bool`): if blur is provided, all trajectories were filtered using
            a gaussian kernel, whose sigma is the value of ``blur``

    Return:
        :obj:`list` of :obj:`tuple`: The relink trajectories.
            Each trajectory is stored in a tuple, (time, positions)
    """
    traj_segments = segment_trajectories(trajectories, window_size, max_frame)
    if debug:
        print("trajectory segmentation finished")
    relinked_segments = []
    if debug:
        i = 0
    for trajs in traj_segments:
        relinked_segments += relink(trajs, dx, dt, blur, blur_velocity, cost=cost, debug=debug)
        if debug:
            i += 1
            message =  f"relink finished for the {i}th trajectory segment, "
            message += f"({(i + 1) / len(traj_segments) * 100:.2f} %)"
            print(message)
    return relinked_segments
