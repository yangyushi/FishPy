from . import utility
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.spatial.transform import Rotation as R


def collect_dynamic_cavagna_style(trajectories, frame, com, next_com, spd_cutoff):
    positions, velocities = [], []
    matched_pair = []
    for t in trajectories:
        if (frame in t.time) and (frame + 1 in t.time):
            ti1 = np.where(t.time == frame)[0][0]
            ti2 = np.where(t.time == frame+1)[0][0]
            matched_pair.append((t.positions[ti1], t.positions[ti2]))  # shape: (n, 2, dim) 
    matched_pair = np.array(matched_pair)
    matched_pair = np.moveaxis(matched_pair, 1, 0)
    dila_mat, rot_mat = utility.get_best_dilatation_rotation(*matched_pair)
    for t in trajectories:
        if (frame in t.time) and (frame+1 in t.time):
            index = np.where(t.time == frame)[0][0]
            next_index = np.where(t.time == frame + 1)[0][0]
            future = t.positions[next_index] - next_com
            current = (t.positions[index] - com)
            current = (current @ dila_mat) @ rot_mat
            dp = future - current
            dt = t.time[next_index] - t.time[index]
            positions.append(t.positions[index])
            velocities.append(dp/dt)
    return np.array(positions), np.array(velocities)

def collect_dynamic(trajectories, frame, spd_cutoff):
    """return the positions and UNIT speed vectors"""
    positions, velocities = [], []
    for t in trajectories:
        if (frame in t.time) and (frame+1 in t.time):
            index = np.where(t.time == frame)[0][0]
            next_index = np.where(t.time == frame+1)[0][0]
            dp = t.positions[next_index] - t.positions[index]
            dt = t.time[next_index] - t.time[index]
            assert dt == 1
            positions.append(t.positions[index])
            spd_norm = np.linalg.norm(dp / dt)
            if spd_norm > spd_cutoff:
                spd_unit = dp / dt #/ spd_norm
                velocities.append(spd_unit)
            else:
                velocities.append(np.zeros(2))

    return np.array(positions), np.array(velocities)

def collect_my_dynamic(trajectories, frame, com, next_com, spd_cutoff):
    positions, velocities = [], []
    for t in trajectories:
        if (frame in t.time) and (frame+1 in t.time):
            index = np.where(t.time == frame)[0][0]
            next_index = np.where(t.time == frame+1)[0][0]
            future = t.positions[next_index] - next_com
            current = t.positions[index] - com
            dp = future - current
            dt = t.time[next_index] - t.time[index]
            assert dt == 1
            spd_norm = np.linalg.norm(dp/dt)
            if spd_norm > spd_cutoff:
                velocities.append(dp/dt)
            else:
                velocities.append(np.zeros(2))
            positions.append(t.positions[index])
    return np.array(positions), np.array(velocities)

def vanilla_cvv(trajectories, bins, frame_range, spd_cutoff=1e-10):
    """
    vanilla velocity correlation function
    Cvv = sum (vi·vj / |vi||vj|)
    trajs: (individual, time_points, dims)
    if speed is smaller than spd_cutoff, the individual is considered as not moving
    """
    cvvs = [[] for _ in bins[:-1]]
    for frame in frame_range:
        cvv = []
        pos, spd = collect_dynamic(trajectories, frame, spd_cutoff)
        if len(pos) == 0:
            continue
        dists = np.triu(squareform(pdist(pos)))
        dists[dists==0] = np.nan
        for i, b in enumerate(bins[:-1]):
            db = bins[i+1] - b
            index_1, index_2 = np.where(np.abs(dists - b - db/2) < db)
            print(index_1, index_2)
            if len(index_1) > 1:
                corr = np.mean([
                    (spd[i1] / np.linalg.norm(spd[i1])).dot(
                     spd[i2] / np.linalg.norm(spd[i2]))
                 for i1, i2 in zip(index_1, index_2)], axis=0)
            else:
                corr = np.nan
            cvvs[i].append(corr)
    return cvvs

def my_cvv(trajectories, bins, frame_range, good_frames=None, spd_cutoff=1e-10):
    """
    vanilla velocity correlation function
    Cvv = sum (vi·vj / |vi||vj|)
    trajs: (individual, time_points, dims)
    if speed is smaller than spd_cutoff, the individual is considered as not moving
    """
    cvvs = [[] for _ in bins[:-1]]
    gce = utility.GCE(trajectories, good_frames)
    coms = gce.centres
    for j, frame in enumerate(frame_range[:-1]):
        com = coms[frame]
        next_com = coms[frame+1]
        if isinstance(com, type(None)) or isinstance(next_com, type(None)):
            continue
        cvv = []
        pos, spd = collect_my_dynamic(trajectories, frame, com, next_com, spd_cutoff)
        if len(pos) == 0:
            continue
        dists = np.triu(squareform(pdist(pos)), k=1)
        for i, b in enumerate(bins[:-1]):
            db = bins[i+1] - b
            index_1, index_2 = np.where(np.abs(dists - b - db/2) < db)
            if len(index_1) > 1:
                corr = np.mean([
                    (spd[i1] / np.linalg.norm(spd[i1])).dot(
                    (spd[i2] / np.linalg.norm(spd[i2])))
                    for i1, i2 in zip(index_1, index_2)], axis=0)
            else:
                corr = np.nan
            cvvs[i].append(corr)
    return cvvs

def cavagna_cvv(trajectories, bins, frame_range, good_frames=None, spd_cutoff=1e-10):
    """
    vanilla velocity correlation function
    Cvv = sum (vi·vj / |vi||vj|)
    trajs: (individual, time_points, dims)
    if speed is smaller than spd_cutoff, the individual is considered as not moving
    """
    cvvs = [[] for _ in bins[:-1]]
    gce = utility.GCE(trajectories, good_frames)
    coms = gce.centres
    for frame in frame_range[:-1]:
        com = coms[frame]
        next_com = coms[frame+1]

        if isinstance(com, type(None)) or isinstance(next_com, type(None)):
            continue
        cvv = []
        pos, spd = collect_dynamic_cavagna_style(trajectories, frame, com, next_com, spd_cutoff)
        spd /= np.sqrt(np.mean([spd[i] @ spd[i] for i in range(spd.shape[0])] , 0))
        #spd /= np.sqrt(np.mean([spd[i] @ spd[i] for i in range(spd.shape[0])] , 0))
        if len(pos) == 0:
            continue
        dists = np.triu(squareform(pdist(pos)))
        dists[dists==0] = np.nan
        for i, b in enumerate(bins[:-1]):
            db = bins[i+1] - b
            index_1, index_2 = np.where(np.abs(dists - b - db/2) < db)
            if len(index_1) > 1:
                corr = np.mean([spd[i1]/np.linalg.norm(spd[i1]) @ spd[i2]/np.linalg.norm(spd[i2]) for i1, i2 in zip(index_1, index_2)], axis=0)
            else:
                corr = np.nan
            cvvs[i].append(corr)
        #cvvs.append(cvv)
    return cvvs

def vanilla_vs(trajectories, bins, frame_range, spd_cutoff=1e-10):
    """get the mean of orientational vector"""
    v2s = [[] for _ in bins[:-1]]
    for frame in frame_range:
        pos, spd = collect_dynamic(trajectories, frame, spd_cutoff)
        if len(pos) == 0:
            continue
        dists = np.triu(squareform(pdist(pos)))
        dists[dists==0] = np.nan
        for i, b in enumerate(bins[:-1]):
            db = bins[i+1] - b
            index_1, index_2 = np.where(np.abs(dists - b - db/2) < db)
            if len(index_1) > 1:
                v = np.mean([spd[i] / np.linalg.norm(spd[i]) for i in np.hstack([index_1, index_2])], axis=0)
            else:
                v = np.nan
            v2s[i].append(v)
    return v2s

def cavagna_vs(trajectories, bins, frame_range, spd_cutoff=1e-10):
    """get the mean of orientational vector"""
    vs = [[] for _ in bins[:-1]]
    for frame in frame_range:
        com = get_com(trajectories, frame, lock=1)
        next_com = get_com(trajectories, frame+1, lock=-1)

        if isinstance(com, type(None)) or isinstance(next_com, type(None)):
            continue
        cvv = []
        pos, spd = collect_dynamic_cavagna_style(trajectories, frame, com, next_com, spd_cutoff)
        spd /= np.sqrt(np.mean([spd[i] @ spd[i] for i in range(spd.shape[0])] , 0))
        #spd /= np.sqrt(np.mean([spd[i] @ spd[i] for i in range(spd.shape[0])] , 0))
        if len(pos) == 0:
            continue
        dists = np.triu(squareform(pdist(pos)))
        dists[dists==0] = np.nan
        for i, b in enumerate(bins[:-1]):
            db = bins[i+1] - b
            index_1, index_2 = np.where(np.abs(dists - b - db/2) < db)
            if len(index_1) > 1:
                v = np.mean([spd[i] / np.linalg.norm(spd[i]) for i in np.hstack([index_1, index_2])], axis=0)
            else:
                v = np.nan
            vs[i].append(v)
    return vs
