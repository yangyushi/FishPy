#!/usr/bin/env python3
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import pickle
import matplotlib.pyplot as plt
from numba import jit
import trackpy as tp
from typing import List
from .nrook import solve_nrook


class Trajectory():
    def __init__(self, time, positions, blur=None):
        assert len(time) == len(positions), "Time points do not match the position number"
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
        self.spd_start = (self.positions[1] - self.positions[0]) / (self.time[1] - self.time[0])
        self.spd_end = (self.positions[-1] - self.positions[-2]) / (self.time[-1] - self.time[-2])

    def predict(self, t):
        """
        predict the position of the particle at time t
        """
        assert t > self.t_end, "We predict the future, not the past"
        pos_predict = self.p_end + self.spd_end * (t - self.t_end)
        return pos_predict

    def __len__(self):
        return len(self.positions)

    def __repr__(self):
        return f"trajectory@{id(self):x}"

    def __str__(self):
        return f"trajectory@{id(self):x}"

    def __add__(self, another_traj):
        """
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

    def break_into_two(self, time_index):
        p1 = self.positions[:time_index]
        t1 = self.time[:time_index]
        p2 = self.positions[time_index:]
        t2 = self.time[time_index:]
        if (len(t1) > 2) and (len(t2) > 2):
            return [Trajectory(t1, p1), Trajectory(t2, p2)]
        else:
            return [self]  # do not break if trying to break tail


class Manager():
    def __init__(self, trajs):
        """
        trajs is a list of trajectories
        trajectory is a list contain two parts, time & positions
        one fish has one trajectory
        """
        self.trajs = trajs

    @jit
    def break_trajectories(self, search_range):
        for i, traj in enumerate(self.trajs):
            for another_traj in self.trajs[i + 1:]:
                dists = cdist(traj.positions, another_traj.positions)
                if np.min(dists) < search_range:
                    dists[dists > search_range] = 0
                    candidates = np.array(dists.nonzero()).T
                    for c in candidates:
                        if traj.time[c[0]] == another_traj.time[c[1]]:
                            breaking_result = self.break_trajectory(i, c[0])
                            if breaking_result:
                                self.break_trajectories(search_range)

    def break_trajectory(self, traj_index, time_point):
        traj_to_break = self.trajs.pop(traj_index)
        new_trajs = traj_to_break.break_into_two(time_point)
        self.trajs += new_trajs
        if len(new_trajs) == 2:
            return True
        else:
            return False

    def as_labels(self):
        """
        labels are a list containing labels of fish at every frame
        labels = [frame_0, ... frame_i, ... (total number of frames)]
        frame_0 = [label_1, label_2, ... label_i, ... (total number of fish in this frame)]
        label_i is a number
        """
        max_frame = np.max(np.hstack([t.time for t in self.trajs]))
        labels, positions = [], []
        for i in range(max_frame + 1):
            l, p = [], []
            for j, traj in enumerate(self.trajs):
                if i in traj.time:
                    l.append(j)
                    pos = traj.positions[np.where(traj.time == i)]
                    p.append(np.squeeze(pos))
            labels.append(np.array(l))
            positions.append(np.array(p))
        return labels, positions

    def relink(self, dist_threshold, time_threshold):
        link_net = LinkingNet(self.trajs, dist_threshold, time_threshold)
        best_link = link_net.get_best_network()
        new_trajs = []
        to_remove = []
        for link in best_link:
            traj_early, traj_late = link.traj_1, link.traj_2
            new_trajs.append(traj_late + traj_early)
            to_remove.append(traj_early)
            to_remove.append(traj_late)
        for t in to_remove:
            self.trajs.remove(t)
        for t in new_trajs:
            self.trajs.append(t)

    def __len__(self):
        return len(self.trajs)

    def __repr__(self):
        return f'trajectory manager @ {id(self):x}'

    def __str__(self):
        return f'trajectory manager: {len(self)} trajectories'


class ActiveLinker():
    def __init__(self, search_range):
        self.search_range = search_range
        self.labels = None
        self.trajectories = None

    def link(self, frames):
        self.labels = self.__get_labels(frames)
        self.trajectories = self.__get_trajectories(self.labels, frames)
        return self.trajectories

    def __predict(self, pos, prev=None):
        if isinstance(prev, type(None)):
            return pos
        else:
            return 2 * pos - prev

    def __get_link_f3(self, f0, f1, f2, links=None):
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
            costs = []
            for l1, p1 in zip(labels_1, candidates_1):
                if l1 not in [l[1] for l in links + new_links]:
                    dist_2 = cdist([self.__predict(p1, p0)], f2)[0]
                    costs.append(np.min(dist_2))
                else:
                    costs.append(np.inf)
            if len(costs) > 0:
                if np.min(costs) < self.search_range * 2:
                    i1 = labels_1[np.argmin(costs)]
                    new_links.append((i0, i1))
        return new_links

    def __get_link_f4(self, fp, f0, f1, f2, old_links):
        new_links = []
        for ip, i0 in old_links:
            p0 = f0[i0]
            dist_1 = cdist([self.__predict(p0, fp[ip])], f1)[0]
            candidates_1 = f1[dist_1 < self.search_range]
            labels_1 = np.arange(len(f1))  # indices of frame #1
            labels_1 = labels_1[dist_1 < self.search_range]
            costs = []
            for l1, p1 in zip(labels_1, candidates_1):
                if l1 not in [l[1] for l in new_links]:
                    dist_2 = cdist([self.__predict(p1, p0)], f2)[0]
                    costs.append(np.min(dist_2))
                else:
                    costs.append(np.inf)
            if (len(costs) > 0) and (np.min(costs) < self.search_range * 2):
                i1 = labels_1[np.argmin(costs)]
                new_links.append((i0, i1))
        new_links += self.__get_link_f3(f0, f1, f2, new_links)
        return new_links

    def __get_links(self, fp, f0, f1, f2, links):
        """
        Get links in two successive frames
        :param fp: previous frame, (3, n) array
        :param f0: current frame, (3, n) array
        :param f1: next frame, (3, n) array
        :param f2: second next frame, (3, n) array
        :param links: link for particles between fp to f0
        :return: link from f0 to f1
        """
        if isinstance(fp, type(None)):
            return self.__get_link_f3(f0, f1, f2)
        else:
            return self.__get_link_f4(fp, f0, f1, f2, links)

    def __get_all_links(self, frames):
        """
        Get all possible links between different frames
        """
        frame_num = len(frames)
        links = None
        for n in range(frame_num - 2):
            if n == 0:
                fp, f0, f1, f2 = None, frames[n], frames[n+1], frames[n+2]  # fp = previous frame
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

    def __get_trajectories(self, labels, frames):
        """
        trajectory = [time_points, positions]
        plot3d(*positions, time_points) --> show the trajectory in 3D
        shape of centres: (N x dim)
        """
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


class TrackpyLinker():
    def __init__(self, max_movement, memory=0, max_subnet_size=30, **kwargs):
        """
        unit of the movement is pixel
        """
        self.max_movement = max_movement
        self.memory = memory
        self.max_subnet_size = max_subnet_size
        self.kwargs = kwargs

    @staticmethod
    def _check_input(positions, time_points, labels):
        """
        make sure the input is proper
        and sequence in time_points are ordered
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
        positions: (time, number_of_individuals, dimensions)
        time_points: (time, )
        labels: (time, number_of_individuals)
        * if labels were given, the returned trajecotory will have a 'label' attribute
          which specifies the label values of the individual in different frames
        * time_points may not be continues
        * The unit of time is NOT converted, do the conversion before running

        labels: [(frame_index, [labels, ... ]), ...]
        """
        if isinstance(time_points, type(None)):
            time_points = np.arange(len(positions))
        pos, time, labels = self._check_input(positions, time_points, labels)
        tp.linking.Linker.MAX_SUB_NET_SIZE = self.max_subnet_size
        link_result = tp.link_iter(pos, search_range=self.max_movement, memory=self.memory, **self.kwargs)
        return self.__get_trajectories(list(link_result), pos, time, labels)


def sort_trajectories(trajectories: List['Trajectory']) -> List['Trajectory']:
    start_time_points = np.empty(len(trajectories))
    for i, traj in enumerate(trajectories):
        start_time_points[i] = traj.t_start
    sorted_indices = np.argsort(start_time_points)
    return [trajectories[si] for si in sorted_indices]


def build_dist_matrix(trajs_sorted: List['Trajectory'], dt: int, dx: float) -> np.ndarray:
    """
    :param dt: if the last time point  of trajectory #1 + dt > first time point of trajectory #2, consider a link being possible
    :param dx: if within dt, the distance between trajectory #1's prediction and trajectory #2's first point is smaller than dx, assign a link
    :return: a matrix records the distance (cost) of the link, if such link is possible
    todo: this funciton needs test
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


def reduce_network(network: np.ndarray) -> List[np.ndarray]:
    """
    network is compose of [(i_1, j_1), (i_2, j_2) ...] links
    the network is *sorted* with i_n > ... > i_2 > i_1
    example: input  ([0, 1], [1, 3], [3, 5], [4, 5], [6, 7], [7, 8])
             output [(0, 1, 3, 5), (4, 5), (6, 7, 8)]
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
    return np.array(reduced)


def choose_network(distance_matrix: np.ndarray, networks: np.ndarray) -> List[np.ndarray]:
    distances = []
    for network in networks:
        costs = distance_matrix[tuple(network.T)]
        dist_total = distance_matrix[network].sum()
        distances.append(dist_total)
    best_network = networks[np.argmin(distances)]
    return reduce_network(best_network)


def apply_network(trajectories: List['Trajectory'], network: List[np.ndarray]) -> List['Trajectory']:
    new_trajs = []
    to_modify = np.hstack(network)
    for link in network:
        to_sum = []
        for i, t in enumerate(trajectories):
            if i in link: to_sum.append(t)
            if len(to_sum) == len(link): break
        for t in to_sum[1:]:
            to_sum[0] = to_sum[0] + t
        new_trajs.append(to_sum[0])
    for i, t in enumerate(trajectories):
        if i not in to_modify:
            new_trajs.append(t)
    return new_trajs


def relink(trajectories, dist_threshold, time_threshold, blur=None, pos_key='position', time_key='time'):
    trajs = [
        Trajectory(t[time_key], t[pos_key], blur=blur) for t in trajectories if len(t[time_key]) > 1
    ]
    trajs_ordered = sort_trajectories(trajs)
    dist_mat = build_dist_matrix(trajs_ordered, dx=dist_threshold, dt=time_threshold)
    networks = solve_nrook(dist_mat.astype(bool))
    if len(networks[0]) == 0:
        return trajectories
    best = choose_network(dist_mat, networks)
    new_trajs = apply_network(trajs_ordered, best)
    return [{'time': t.time, 'position': t.positions} for t in new_trajs]


if __name__ == "__main__":
    with open('broken_manager.pkl', 'rb') as f:
        manager = pickle.load(f)
    print('before relink', manager)

    trajs = []
    ll = np.arange(20, 800, 20)
    dt = 10
    for l in ll:
        manager.relink(l, dt)
        print(f'after relink (dist={l}), {manager}')
        trajs.append(len(manager))
    plt.plot(ll, trajs, '-o', color='tomato', markeredgecolor='k')
    plt.show()
