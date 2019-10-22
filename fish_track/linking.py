#!/usr/bin/env python3
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import pickle
import matplotlib.pyplot as plt
from numba import jit
import trackpy as tp
from nrook import solve_nrook


class Trajectory():
    def __init__(self, time, positions, blur=None):
        assert len(time) == len(positions), "Time points do not match the position number"
        self.time = time
        self.positions = positions
        self.length = len(time)
        if blur:
            self.positions = ndimage.gaussian_filter1d(self.positions, blur, axis=0)
        self.p_start = self.positions[0]
        self.p_end = self.positions[-1]
        self.t_start = self.time[0]
        self.t_end = self.time[-1]
        self.spd_start = (self.positions[1] - self.positions[0]) / (self.time[1] - self.time[0])
        self.spd_end = (self.positions[-1] - self.positions[-2]) / (self.time[-1] - self.time[-2])

    def predict(self, t):
        """
        predict the position and speed of the particle at time t
        acceleration is ignored --> spd_predict = self.spd_end
        """
        assert t > self.t_end, "We predict the future, not the past"
        pos_predict = self.p_end + self.spd_end * (t - self.t_end)
        spd_predict = self.spd_end#[-1]
        return pos_predict, spd_predict

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


class Link():
    def __init__(self, traj_1, traj_2):
        assert type(traj_1) == Trajectory, "Link is only for trajectories, not %s" % type(traj_1)
        assert type(traj_2) == Trajectory, "Link is only for trajectories, not %s" % type(traj_2)
        self.traj_1, self.traj_2 = traj_1, traj_2
        self.pos_dist = np.linalg.norm(traj_1.predict(traj_2.t_start) - traj_2.p_start)
        #self.pos_dist = np.linalg.norm(traj_1.p_end - traj_2.p_start)
        self.spd_dist = np.linalg.norm(traj_1.spd_end - traj_2.spd_end) * (traj_2.t_start - traj_1.t_end)
        #self.spd_dist = np.linalg.norm(traj_1.spd_end - traj_2.spd_start) * (traj_2.t_start - traj_1.t_end)
        self.dist = np.sqrt(self.pos_dist**2 + self.spd_dist**2)

    def __and__(self, link):
        assert type(link) == Link, "logic and is only for Link instance"
        c11 = link.traj_1 is self.traj_1
        c12 = link.traj_1 is self.traj_2
        c21 = link.traj_2 is self.traj_1
        c22 = link.traj_2 is self.traj_2
        return c11 or c12 or c21 or c22

    def conflict(self, links):
        conf = []
        for l in links:
            if self & l:
                conf.append(l)
        return conf


class Network():
    def __init__(self):
        "collection of links"
        self.net = []
        self.length = 0
        self.index = 0

    def get_cost(self):
        cost = 0
        for link in self.net:
            cost += link.dist
        return cost

    def __add__(self, link):
        assert type(link) == Link, "Only links can be added into the network"
        self.net.append(link)
        self.length += 1
        return self

    def __iadd__(self, link):
        self.__add__(link)
        return self

    def __sub__(self, link):
        assert link in self.net, "Only the links within the network can be removed"
        self.net.remove(link)
        self.length -= 1
        return self

    def __isub__(self, link):
        self.__sub__(link)
        return self

    def __len__(self):
        return self.length

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        link = self.net[self.index]
        self.index += 1
        return link

    def __eq__(self, network):
        assert type(network) == Network, f"{type(network)} can not be compared with a network"
        if len(self) != len(network):
            return False
        for link in self.net:
            if link not in network.net:
                return False
        for link in network.net:
            if link not in self.net:
                return False
        return True

    def __lt__(self, network):
        return len(self) < len(network)

    def __gt__(self, network):
        return len(self) > len(network)

    def __le__(self, network):
        return len(self) <= len(network)

    def __ge__(self, network):
        return len(self) >= len(network)


    def __contains__(self, network):
        assert type(network) == Network, f"{type(network)} can not be contained in a network"
        for link in network:
            if link not in self.net:
                return False
        return True


class LinkingNet():
    def __init__(self, trajs, dist_threshold, time_threshold):
        self.trajs = trajs
        self.dist_threshold = dist_threshold
        self.time_threshold = time_threshold
        self.links = self.get_possible_links()

    def get_possible_links(self):
        """
        get links that link trajectories together
        if their 'distance' is below the dist_threshold
        also if their time interval is smaller than time_threshold
        """
        possible_links = []  # [trajectory_early, trajectory_late, distance]
        for t1 in self.trajs:
            for t2 in self.trajs:
                if t1 is t2:
                    continue
                dt = t2.t_start - t1.t_end
                if (dt > 0) and (dt < self.time_threshold):
                    x1p, v1p = t1.predict(t2.t_start)
                    dist_1 = np.linalg.norm(x1p - t2.p_start) ** 2
                    dist_2 = (np.linalg.norm(v1p - t2.spd_start) * (t2.t_start - t1.t_end)) ** 2
                    dist = np.sqrt(dist_1 + dist_2)
                    if dist < self.dist_threshold:
                        possible_links.append(Link(t1, t2))
        if len(possible_links) > 20:
            print(f"{len(possible_links)} possible links found!")
        return possible_links

    def get_networks(self, links=None, networks=None, level=0):
        """
        generate networks from possible links
        one trajectory can only be used once in each network
        considering following possible links
        """
        if not networks:
            networks = []
        used = []
        networks.append(Network())
        current_network = networks[-1]
        if not links:
            links = self.links
        for link in links:
            conf = link.conflict(used)
            if conf: # branching without conflicting element
                self.get_networks([l for l in links if l not in conf], networks, level=level+1)
            else:
                current_network += link
                used += [link]
        if level == 0: # only return something in the top recursion
            return networks

    def get_best_network(self):
        all_networks = self.get_networks()
        if len(all_networks) > 1000:
            print(f"{len(all_networks)} networks were found")
        good_networks = self.merge_networks(all_networks)
        costs = [np.sum([net.get_cost() for net in good_networks])]
        best_network = all_networks[np.argmin(costs)]
        return best_network

    def merge_networks(self, networks):
        for n1 in networks:
            for n2 in networks:
                if (n1 < n2):
                    if n1 in n2:
                        networks.remove(n1)
                        self.merge_networks(networks)
        return networks


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


def relink(trajectories, dist_threshold, time_threshold, pos_key='position', time_key='time'):
    traj_objs = [
        Trajectory(t[time_key], t[pos_key]) for t in trajectories if len(t[time_key]) > 2
    ]
    link_net = LinkingNet(traj_objs, dist_threshold, time_threshold)
    best_link = link_net.get_best_network()
    new_trajs = []
    to_remove = []
    for link in best_link:
        traj_early, traj_late = link.traj_1, link.traj_2
        new_trajs.append(traj_late + traj_early)
        to_remove.append(traj_early)
        to_remove.append(traj_late)
    for t in to_remove:
        traj_objs.remove(t)
    for t in new_trajs:
        traj_objs.append(t)
    return [{'time': t.time, 'position': t.positions} for t in traj_objs]


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
