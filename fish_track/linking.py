#!/usr/bin/env python3
import os
import pickle
import numpy as np
import trackpy as tp
from numba import njit
from typing import List
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from joblib import delayed, Parallel
from numba.typed import List as nList
from scipy.spatial.distance import cdist
from .nrook import solve_nrook, solve_nrook_dense


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
    def __init__(self, time, positions, blur=None, velocities=None):
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
            self.v_end = (self.positions[-1] - self.positions[-2]) / (self.time[-1] - self.time[-2])

    def predict(self, t):
        """
        predict the position of the particle at time t
        """
        assert t > self.t_end, "We predict the future, not the past"
        pos_predict = self.p_end + self.v_end * (t - self.t_end)
        return pos_predict

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

    def break_into_two(self, time_index):
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
            frames (:obj:`numpy.ndarray`): the positions of particles in different frames from the experimental data

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
            fp (np.ndarray): previous frame, (dim, n)
            f0 (np.ndarray): current frame, (dim, n)
            f1 (np.ndarray): next frame, (dim, n)
            f2 (np.ndarray): second next frame, (dim, n)
            links (list): link for particles between fp to f0

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
                    new_label = max(label_set) + 1
                    new_labels[s] = new_label
                    label_set.add(new_label)
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
        [labels_numba.append(l) for l in labels]
        [frames_numba.append(l) for l in frames]
        trajectories = []
        for target in range(max_value):  # find the trajectory for every label
            result = get_trajectory(labels_numba, frames_numba, target)
            time, positions = result
            trajectories.append([
                np.array(time),
                np.array(positions)
            ])
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
        if isinstance(labels, type(None)):
            traj = {'time': [], 'position': []}
        else:
            traj = {'time': [], 'position': [], 'label': []}
        for frame in link_result:
            frame_index, link_labels = frame
            if value in link_labels:
                number_index = link_labels.index(value)
                traj['time'].append(time_points[frame_index])
                traj['position'].append(positions[frame_index][number_index])
                if 'label' in traj:
                    current_label = labels[frame_index][link_labels.index(value)]
                    traj['label'].append(current_label)
        traj['time'] = np.array(traj['time'])
        traj['position'] = np.array(traj['position'])
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
        trajectories (List[Trajectory]): a collection of :class:`Trajectory`

    Return:
        List[Trajectory]: the sorted trajectories
    """
    start_time_points = np.empty(len(trajectories))
    for i, traj in enumerate(trajectories):
        start_time_points[i] = traj.t_start
    sorted_indices = np.argsort(start_time_points)
    return [trajectories[si] for si in sorted_indices]


def build_dist_matrix(trajs_sorted, dt, dx):
    """
    Args:
        trajs_sorted (list): trajectories obtained from :meth:`sort_trajectories`
        dt (int): if the time[-1] of traj #1 + dt > time[0] of traj #2
                  consider a link being possible
        dx (float): if within dt, the distance between
                    traj #1's prediction and
                    traj #2's first point
                    is smaller than dx, assign a link

    Return:
        np.ndarray: the distance (cost) of the link, if such link is possible
    """
    traj_num = len(trajs_sorted)
    dist_matrix = np.zeros((traj_num, traj_num), dtype=float)
    for i, traj_1 in enumerate(trajs_sorted):
        for j, traj_2 in enumerate(trajs_sorted[i+1:]):
            if (traj_2.t_start - traj_1.t_end) > dt:
                break
            elif traj_2.t_start > traj_1.t_end:
                distance = np.linalg.norm(traj_1.predict(traj_2.t_start) - traj_2.p_start)
                if distance < dx:
                    dist_matrix[i, i+j+1] = distance
    return dist_matrix


def squeeze_sparse(array: np.array) -> np.array:
    """
    Given the indices in a row or column from a sparse matrix
    Remove the blank rows/columns

    .. code-block::

        array[i + 1] >= array[i]
    """
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


def build_dist_matrix_sparse(trajs_sorted, dt, dx):
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

    Return:
        tuple: (distance matrix (scipy.sparse.coo_matrix), row_map, col_map)
               The ``row_map`` and ``col_map`` are used to map
               from squeezed distance matrix to origional distance matrix
    """
    values, rows, cols = [], [], []
    for i, traj_1 in enumerate(trajs_sorted):
        for j, traj_2 in enumerate(trajs_sorted[i+1:]):
            if (traj_2.t_start - traj_1.t_end) > dt:
                break
            elif traj_2.t_start > traj_1.t_end:
                distance = np.linalg.norm(traj_1.predict(traj_2.t_start) - traj_2.p_start)
                if distance < dx:
                    rows.append(i)
                    cols.append(i+j+1)
                    values.append(distance)
    if len(values) == 0:
        return np.empty(0), {}, {}
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    values = np.array(values, dtype=float)
    rows_sqz = squeeze_sparse(rows)
    cols_sqz = squeeze_sparse(cols)
    row_map = {r1 : r0 for r0, r1 in zip(rows, rows_sqz)}  # map from squeezed to origional
    col_map = {c1 : c0 for c0, c1 in zip(cols, cols_sqz)}
    larger = max(max(rows_sqz), max(cols_sqz)) + 1
    dist_mat = coo_matrix((values, (rows_sqz, cols_sqz)), shape=(larger, larger), dtype=float).toarray()
    return dist_mat, row_map, col_map


def reduce_network(network) -> List[np.ndarray]:
    """
    Network is compose of [(i_1, j_1), (i_2, j_2) ...] links
    The network is *sorted* with i_n > ... > i_2 > i_1

    Args:
        network (np.ndarray): a collection of links

    Return:
        np.ndarray: the reduced network

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
        ll = np.where(link[-1] == network[:, 0])[0]
        while len(ll) > 0:
            to_skip.append(ll)
            new_link = np.hstack((new_link, network[ll[0], 1]))
            ll = np.where(new_link[-1] == network[:, 0])[0]
        reduced.append(new_link)
    return reduced


def choose_network(distance_matrix: np.ndarray, networks: np.ndarray) -> np.ndarray:
    distances = []
    for network in networks:
        costs = distance_matrix[tuple(network.T)]
        dist_total = distance_matrix[network].sum()
        distances.append(dist_total)
    return networks[np.argmin(distances)]


def apply_network(trajectories, network):
    """
    Args:
        trajectories (List[Trajectory]):
        network (List[np.ndarray]):

    Return:
        List[Trajectory]: a collection of trajectories
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


def relink_slow(trajectories, dist_threshold, time_threshold, blur=None, pos_key='position', time_key='time'):
    trajs = [
        Trajectory(t[time_key], t[pos_key], blur=blur) for t in trajectories if len(t[time_key]) > 1
    ]
    trajs_ordered = sort_trajectories(trajs)
    dist_mat = build_dist_matrix(trajs_ordered, dx=dist_threshold, dt=time_threshold)
    networks = solve_nrook(dist_mat.astype(bool))
    if len(networks[0]) == 0:
        return trajectories
    best = choose_network(dist_mat, networks)
    reduced = reduce_network(best)
    new_trajs = apply_network(trajs_ordered, reduced)
    return [{'time': t.time, 'position': t.positions} for t in new_trajs]


def relink(trajectories: List, dist_threshold: float, time_threshold: int,
        blur=None, pos_key='position', time_key='time') -> List[dict]:
    """
    Re-link short trajectories into longer
    normal usage: relink(trajs, dx, dt, blur)
    """
    if isinstance(trajectories[0], dict):
        trajs = [
            Trajectory(
                t[time_key], t[pos_key], blur=blur
            ) for t in trajectories if len(t[time_key]) > 1
        ]
    elif isinstance(trajectories[0], list):  # (time, positions)
        trajs = [
            Trajectory(
                t[0], t[1], blur=blur
            ) for t in trajectories if len(t[0]) > 1
        ]
    else:
        return TypeError("Invalid Trajectory Data Type")
    trajs_ordered = sort_trajectories(trajs)

    dist_mat, row_map, col_map = build_dist_matrix_sparse(
        trajs_ordered, dx=dist_threshold, dt=time_threshold
    )

    if len(dist_mat) == 0:
        return trajectories

    max_row = max(row_map.keys())+1
    networks = solve_nrook_dense(dist_mat.astype(bool), max_row=max_row)

    best = choose_network(dist_mat, networks)

    for i, pair in enumerate(best):
        best[i, 0] = row_map[pair[0]]
        best[i, 1] = col_map[pair[1]]

    reduced = reduce_network(best)
    new_trajs = apply_network(trajs_ordered, reduced)

    return [{'time': t.time, 'position': t.positions} for t in new_trajs]


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
