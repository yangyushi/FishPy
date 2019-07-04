#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np
import pickle
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from numba import njit, jit

class Analyser():
    def __init__(self, trajs, origin=None):
        """np.diff --> x(i+1) - x(i)"""
        self.trajs = trajs
        self.frame_nums = list(set(np.hstack([t.time for t in self.trajs]).ravel()))
        if isinstance(origin, type(None)):
            self.origin = self._get_com()  # different origin at each time point
        else:
            self.origin = np.array([origin for _ in self._get_com()])
        self.ndim = len(self.origin[0])
        self.coms = self._get_com()
        self.rot_matrices, self.dila_matrices = self._get_rot_dila_matrices()
        self.rot_vecs = [Rotation.from_dcm(rm).as_rotvec() for rm in self.rot_matrices]
        
    def _get_com(self):
        """calculate the centre of mass at different frames"""
        coms = []
        for frame in self.frame_nums:
            positions = []
            for t in self.trajs:
                if frame in t.time:
                    index = np.argmin(np.abs(t.time - frame))
                    positions.append(t.positions[index])
            coms.append(np.mean(positions, axis=0))
        return np.array(coms)
        
    def get_positions_r(self):
        """get positions relative to the centre of mass of while group"""
        relative_positions = []
        for t in self.trajs:
            time_indices = []
            for time_point in t.time:
                time_index = np.where(self.frame_nums == time_point)[0][0]
                time_indices.append(time_index)
            relative_positions.append(t.positions - self.origin[time_indices])
        return relative_positions
    
    def get_velocities_r(self):
        """get velocities relative to the centre of mass of while group"""
        positions = self.get_positions_r()
        spds = []
        for i, t in enumerate(self.trajs):
            spd = np.diff(positions[i], axis=0) / np.expand_dims(np.diff(t.time), 1)
            spds.append(spd)
        return spd
    
    def get_simple_velocity_fluctuations(self):
        """
        get the velocity fluctions relative to the velocity of centre of mass
        velocity = x(t+1) - x(t)  [in normal frame, not com frame]
        fluctuations = velocity - velocity_com
        """
        fluctuations = []
        for i, t in enumerate(self.trajs):
            time_index = np.where(t.time == self.frame_nums)[0]
            velocity_com = np.diff(self.origin[time_index], axis=0) / np.expand_dims(np.diff(time_index), 1)
            velocities = np.diff(t.positions, axis=0) / np.expand_dims(np.diff(time_index), 1)
            fluctuations.append(velocities - velocity_com)
        return fluctuations
    
    def _get_rot_matrices(self):
        matrices = []
        for i, frame in enumerate(self.frame_nums[:-1]):
            next_frame = self.frame_nums[i+1]
            origin = self.origin[i]

            existing_trajs = [traj for traj in self.trajs if (frame in traj.time) and ((frame + 1) in traj.time)]
            current_points = np.array([t.positions[i] for t in existing_trajs if t.time[i] == frame])
            future_points = np.array([t.positions[i] for t in existing_trajs if t.time[i] == frame + 1])

            print(current_points.shape, future_points.shape)

            assert len(current_points == future_points), "points not matching"
            
            rot_mat = self.get_best_rotation(current_points, future_points)

            matrices.append(rot_mat)
        return matrices
            
    def _get_rot_dila_matrices(self):
        rot_matrices, dila_matrices = [], []
        for i, frame in enumerate(self.frame_nums[:-1]):
            next_frame = self.frame_nums[i+1]
            origin = self.origin[i]

            existing_trajs = [traj for traj in self.trajs if (frame in traj.time) and ((frame + 1) in traj.time)]
            current_points, future_points = [], []
            for traj in existing_trajs:
                current_time_index = np.where(traj.time == frame)
                future_time_index = np.where(traj.time == frame + 1)
                current_points.append(traj.positions[current_time_index])
                future_points.append(traj.positions[future_time_index])
            current_points = np.squeeze(current_points)
            future_points = np.squeeze(future_points)
            
            dila_mat, rot_mat = self.get_best_dilatation_rotation(current_points, future_points)

            rot_matrices.append(rot_mat)
            dila_matrices.append(dila_mat)
        return rot_matrices, dila_matrices
        
            
    def get_best_dilatation_rotation(self, r1, r2, init_guess=None):
        """
        calculate numeratically
        (r1 @ Lambda) @ Rotation = r2, hopefully
        """
        if isinstance(init_guess, type(None)):
            init_guess = np.ones(r1.shape[1])

        def cost(L, r1, r2, self):
            Lambda = np.identity(r1.shape[1]) * np.array(L)
            r1t = r1 @ Lambda
            R = self.get_best_rotation(r1t, r2)
            return np.sum(np.linalg.norm(r2 - r1t @ R))

        result = least_squares(cost, init_guess, args=(r1, r2, self))
        L = np.identity(r1.shape[1]) * np.array(result['x'])
        r1t = r1 @ L
        R = self.get_best_rotation(r1t, r2)
        return L, R
    
    @staticmethod
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
            
    def get_trans_orders(self):
        """
        get translational order parameter at different frame
        in each frame, automatically average over induviduals inside that frame
        """
        orders = []
        for frame in self.frame_nums[:-1]:  # diff --> x(t+1) - x(t)
            velocities = []
            for i, t in enumerate(self.trajs):
                if frame in t.time:
                    index = np.argmin(np.abs(t.time - frame))
                    if index < len(t.time) - 1:  # ignore trajectory if its last time point is frame
                        velocities.append(self.get_velocities(t)[index])
            directions = velocities / np.expand_dims(np.linalg.norm(velocities, axis=1), 1)
            orders.append(np.abs(np.nanmean(directions)))
        return orders
    
    @jit
    def get_rot_orders(self):
        """
        get rotational order parameter at different frame
        in each frame, automatically average over induviduals inside that frame
        since this is in 2D, we know the rotational axis should always be (0, 0, 1)
        """
        orders = []
        for j, frame in enumerate(self.frame_nums[:-1]):  # diff --> x(t+1) - x(t)
            rotations = []
            for i, t in enumerate(self.trajs):
                if frame in t.time:
                    index = np.argmin(np.abs(t.time - frame))
                    if index < len(t.time) - 1:  # ignore trajectory if its last time point is frame
                        #v = self.get_velocities(t)[index]
                        v = (t.positions[index] - t.positions[index - 1]) / (t.time[index] - t.time[index - 1])
                        p = t.positions[index] - self.origin[index]
                        if self.ndim == 2:
                            p = np.hstack([p, 0])
                            rotations.append(np.cross(p, v) @ np.array([0, 0, 1]))
                        else:
                            k = self.rot_vecs[j]
                            rotations.append(np.cross(p, v) @ k)
            rot_directions = rotations / np.linalg.norm(rotations, axis=0)
            orders.append(np.nanmean(rot_directions))
        return orders
    
    @jit
    def get_dila_orders(self):
        orders = []
        positions = self.get_positions_r()
        for fi, frame in enumerate(self.frame_nums[:-1]):  # diff --> x(t+1) - x(t), todo: fi is a bad name!
            dilations, dilations_abs = [], []
            for i, t in enumerate(self.trajs):
                if frame in t.time:
                    index = np.argmin(np.abs(t.time - frame))
                    if index < len(t.time) - 1:  # ignore trajectory if its last time point is frame
                        p = positions[i][index] @ self.rot_matrices[fi]
                        p_1 = positions[i][index + 1]
                        dilations.append(p @ (p_1 - p))
                        dilations_abs.append(np.linalg.norm(p) * np.linalg.norm(p_1 - p))
            dila_directions = np.array(dilations) / np.array(dilations_abs)
            orders.append(np.nanmean(dila_directions))
        return orders   
    
    def get_vf_scale(self, time_index):
        """get the scale factor to make velocity fluctuation dimensionless"""
        frame_index = self.frame_nums[time_index]
    
    def get_velocity_fluctuations(self):
        """
        get the velocity fluctuations of individuals at different frames
        shape of the result:
        [
            [fish_1_frame_1, fish_1_frame_2, ...],
            [fish_2_frame_1, fish_2_frame_2, ...],
            ...
        ]
        The different fish have different length!
        """
        fluctrations = []
        for traj in self.trajs:
            fluctrations.append([])
            for i, position in enumerate(traj.positions[:-1]):
                time_point = traj.time[i]
                time_point_index = self.frame_nums.index(time_point)
                no_trans = position - self.coms[time_point_index]
                no_collective = (no_trans @ self.dila_matrices[time_point_index]) @ self.rot_matrices[time_point_index]
                velocity_fluctration = (
                    traj.positions[i+1] - self.origin[i+1] - no_collective
                ) / (traj.time[i+1] - traj.time[i])
                fluctrations[-1].append(velocity_fluctration)
        return fluctrations
    
    def get_dimless_velocity_fluctuations(self):
        """
        get dimensionless velocity fluctuations
        """
        vfs = self.get_velocity_fluctuations()
        vfs_dimless = vfs.copy()
        max_frame = np.max(self.frame_nums)
        scale_factors = []  # at different frames
        for i in range(max_frame):
            scale = []
            for j, vf in enumerate(vfs):
                if i in self.trajs[j].time[:-1]:  # velocity difference is v_i+1 - v_i
                    time_index = np.argmin(np.abs(self.trajs[j].time - i))
                    scale.append(vf[time_index] @ vf[time_index])
            scale = np.sqrt(np.mean(scale, axis=0))
            for k, vf in enumerate(vfs_dimless):
                if i in self.trajs[k].time[:-1]:
                    tii = np.argmin(np.abs(self.trajs[k].time - i))  # time index i
                    vf[tii] = vf[tii] / scale
        return vfs_dimless
    
    @staticmethod
    def get_velocities(traj):
        velocities = np.diff(traj.positions, axis=0) / np.expand_dims(np.diff(traj.time), 1)
        return velocities


def plot_dila_orders(analyser):
    d_orders = analyser.get_dila_orders()
    plt.subplot(121).plot(d_orders, label='group centre', color='tomato')
    plt.legend(fontsize=12, loc='lower right')
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('time / frame', fontsize=18)
    plt.ylabel('$\Lambda$', fontsize=18)
    plt.subplot(122).hist(d_orders, bins=np.linspace(-1, 1, 50), color='tomato', edgecolor='k')
    plt.xlabel('$\Lambda$', fontsize=18)
    plt.ylabel('PDF', fontsize=18)
    #plt.gcf().set_size_inches(8, 2.5)
    plt.tight_layout()
    plt.show()
