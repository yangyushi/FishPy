#!/usr/bin/env python3
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
import pickle
import matplotlib.pyplot as plt
from numba import jit


def get_trajectory_from_labels(results, centres):
    """
    trajectory = [time_points, positions]
    plot3d(*positions, time_points) --> show the trajectory in 3D
    shape of centres: (N x dim)
    """
    frame_nums = [l[0] for l in results]
    labels = [l[1] for l in results]
    max_value = np.hstack(labels).max()
    trajectories = []
    for i in range(max_value):  # find the trajectory for every label
        time, positions = [], []
        for frame, c in enumerate(centres):
            if i in labels[frame]:
                index = labels[frame].index(i)
                positions.append(c[index])  # (xy, xy2, ..., xyn)
                time.append(frame)
        trajectories.append([np.array(time), np.array(positions)])
    return trajectories


class FishTrajectory():
    def __init__(self, time, positions, blur=None):
        assert len(time) == len(positions), "Time points do not match the position number"
        self.time = time
        self.positions = positions
        self.length = len(time)
        self.blur = blur
        if blur:
            self.positions = ndimage.gaussian_filter1d(self.positions, blur)
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
        spd_predict = self.spd_end[-1]
        return pos_predict, spd_predict

    def __len__(self):
        return len(self.positions)
    
    def __repr__(self):
        return f"trajectory@{id(self):x}"
        
    def __str__(self):
        return f"trajectory@{id(self):x}"
    
    def __add__(self, another_traj):
        assert type(another_traj) == FishTrajectory, "Only Fish Trajectories can be added together"
        if self.t_start <= another_traj.t_end:  # self is earlier
            new_time = np.concatenate([self.time, another_traj.time])
            new_positions = np.concatenate([self.positions, another_traj.positions])
            return FishTrajectory(new_time, new_positions, self.blur)
        elif self.t_end >= another_traj.t_start:  # self is later
            new_time = np.concatenate([another_traj.time, self.time])
            new_positions = np.concatenate([another_traj.positions, self.positions])
            return FishTrajectory(new_time, new_positions, self.blur)
        else:  # there are overlap between time
            return self
        
    def break_into_two(self, time_index):
        p1 = self.positions[:time_index]
        t1 = self.time[:time_index]
        p2 = self.positions[time_index:]
        t2 = self.time[time_index:]
        if (len(t1) > 2) and (len(t2) > 2):
            return [FishTrajectory(t1, p1, self.blur), FishTrajectory(t2, p2, self.blur)]
        else:
            return [self]  # do not break if trying to break tail
        

class FishManager():
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
        

class Trajectories():
    def as_movie():
        """
        organise trajectory frame-by-frame
        """
        pass

    def as_individuals():
        """
        organise trajectory label-by-label (individual-by-individual)
        """
        pass


class Link():
    def __init__(self, traj_1, traj_2):
        assert type(traj_1) == FishTrajectory, "Link is only for trajectories"
        assert type(traj_2) == FishTrajectory, "Link is only for trajectories"
        self.traj_1, self.traj_2 = traj_1, traj_2
        self.pos_dist = np.linalg.norm(traj_1.predict(traj_2.t_start) - traj_2.p_start)
        #self.pos_dist = np.linalg.norm(traj_1.p_end - traj_2.p_start)
        self.spd_dist = np.linalg.norm(traj_1.spd_end - traj_2.spd_end) * (traj_2.t_start - traj_1.t_end)
        #self.spd_dist = np.linalg.norm(traj_1.spd_end - traj_2.spd_start) * (traj_2.t_start - traj_1.t_end)
        self.dist = np.sqrt(self.pos_dist**2 + self.spd_dist**2)

        
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
                dt = t2.t_start - t1.t_end
                if (t1 is not t2) and (dt > 0) and (dt < self.time_threshold):
                    x1p, v1p = t1.predict(t2.t_start)
                    dist_1 = np.linalg.norm(x1p - t2.p_start) ** 2
                    dist_2 = (np.linalg.norm(v1p - t2.spd_start) * (t2.t_start - t1.t_end)) ** 2
                    dist = np.sqrt(dist_1 + dist_2)
                    if dist < self.dist_threshold:
                        possible_links.append(Link(t1, t2))
        return possible_links
    
    def get_networks(self, links=None, networks=None, history=None, level=0):
        """
        generate networks from possible links
        one trajectory can only be used once in each network
        considering following possible links
        """
        if not networks:
            networks = []
        if not history:
            history = []
        used = []
        networks.append(Network())
        current_network = networks[-1]
        if not links:
            links = self.links
        for i, link in enumerate(links):
            if (link.traj_1 in used) or (link.traj_2 in used):
                    # branching without conflicting element
                self.get_networks([l for l in links if l is not link], networks, history, level=level+1)  
            else:
                current_network += link
                used += [link.traj_1, link.traj_2]
                history += [link.traj_2, link.traj_2]
        if links == self.links: # only return something in the top recursion
            return networks
                
    def get_best_network(self):
        all_networks = self.get_networks()
        good_networks = self.merge_networks(all_networks)
        costs = [np.sum([net.get_cost() for net in good_networks])]
        best_network = all_networks[np.argmin(costs)]
        return best_network
        
    def merge_networks(self, networks):
        for n1 in networks:
            for n2 in networks:
                if (n1 is not n2) and (n1 in n2):
                    networks.remove(n1)
                    self.merge_networks(networks)
        return networks


def link(frames, model):
    """
    frames = [positions_1, positions_2, ...]
    positions_1 = [(x1, x2 , ...), (y1, y2, ...), (z1, z2, ...)]
    """
    for i, frame in enumerate(frames[:-3]):
        if i == 0:
            pass  # use the special 2 frame linking algorithm
        elif i == 2:
            pass  # use the special 3 frame linking algorithm
        else:
            pass  # use the standard 4 frame linking algorithm
    return labels


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
