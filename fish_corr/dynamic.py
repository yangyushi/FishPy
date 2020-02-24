#!/usr/bin/env python3
import numpy as np
import pickle
from scipy.stats import binned_statistic
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import tqdm
from . import utility


class Critic():
    """
    Calculate the dynamical order & correlations from a fish movie

        movie : a `Movie` instance from package `fish_track`

            Movie[ f ]             - the positions of all particles in frame f
            Movie.velocity( f )    - the velocities of all particles in frame f
            Movie.label( f )       - the labels of all particles in frame f, same label = same identity
            Movie.trajs[ i ]       - the positions of trajectory i. Here _i_ is the _label_ for these positions
            Movie.indice_pair( f ) - the paried indices of frame _f_ and frame _f+1_
                                     example:
                                         p0, p1 = Movie.indice_pair(f)
                                         Movie[f][p0] & Movie[f+1][p1] correspond to the same particles

        flctn_not : non-translational fluctuation
                    (the collective translation is removed)
        flctn_noi : non-isometric fluctuation
                    (the collective translation & rotaton is removed)
        flctn_nos : non-similar fluctuation
                    (the collective translation & rotation & isotropic scaling is removed)
    """
    def __init__(self, movie):
        self.movie = movie
        self.trajs = movie.trajs
        self.__pos_pair = {}
        self.__GR = {}
        self.__R = {}
        self.__flctn_not = {}
        self.__flctn_noi = {}
        self.__flctn_nos = {}

    def get_position_pair(self, frame):
        """
        return the same points in _frame_ and _frame+1_
        the centres were substrated for both frames
        """
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no position pair in frame {frame}")
        elif frame in self.__pos_pair.keys():
            return self.__pos_pair[frame]
        else:
            idx_0, idx_1 = self.movie.indice_pair(frame)
            if len(idx_0) > 0:
                pos_0 = self.movie[frame+0][idx_0]
                pos_1 = self.movie[frame+1][idx_1]
                r0 = pos_0 - np.mean(pos_0, axis=0)
                r1 = pos_1 - np.mean(pos_1, axis=0)
                self.__pos_pair.update({frame: (r0, r1)})
            else:
                r0 = np.empty((0, 3))
                r1 = np.empty((0, 3))
            return r0, r1

    def get_isometry(self, frame):
        """
        return the best isometric transformation (rotation)
        to transform points in _frame_ to _frame+1_
        """
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no isometry in frame {frame}")
        elif frame in self.__R.keys():
            return self.__R[frame]
        else:
            r0, r1 = self.get_position_pair(frame)
            R = utility.get_best_rotation(r0, r1)
            self.__R.update({frame: R})
            return R

    def get_similarity(self, frame):
        """
        return the best similar transformation (rotation + dilation)
        to transform points in _frame_ to _frame+1_
        """
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no similarity in frame {frame}")
        elif frame in self.__GR.keys():
            return self.__GR[frame]
        else:
            r0, r1 = self.get_position_pair(frame)
            # Kabsch algorithm
            G, R = utility.get_best_dilatation_rotation(r0, r1)
            self.__GR.update({frame: (G, R)})
            return G, R

    def get_orders(self, start=0, stop=None, size_threshold=5, report=False):
        """
        return nan for one frame if the matched pair in that frame is smaller than size_threshold
        """
        if not stop:
            stop = len(self.movie) - 1

        t_orders, r_orders, d_orders = [], [], []

        if report:
            frames = tqdm(range(start, stop+1))
        else:
            frames = range(start, stop+1)

        for frame in frames:
            idx_0, idx_1 = self.movie.indice_pair(frame)

            if len(idx_0) < size_threshold:
                t_orders.append(np.nan)
                r_orders.append(np.nan)
                d_orders.append(np.nan)
                continue

            r0, r1 = self.get_position_pair(frame)
            velocities = self.movie.velocity(frame)[idx_0]

            G, R = self.get_similarity(frame)

            # calculate translational order
            directions = velocities.T / np.linalg.norm(velocities, axis=1)
            t_orders.append(np.linalg.norm(np.mean(directions, axis=1)))

            # calculate rotational order
            K = Rotation.from_dcm(R).as_rotvec()
            K = K / np.linalg.norm(K)
            y_pp = r0 - K * (r0 @ np.vstack(K))  # r0 projected to plane ⊥ K, (n, 3)
            ang_mom = np.cross(y_pp, velocities)  # angular momentum, shape (n, 3)
            ang_unit = ang_mom / np.vstack(np.linalg.norm(ang_mom, axis=1))
            r_orders.append(np.mean(ang_unit @ np.vstack(K)))

            # calculate dilational order
            D = (r0 @ R) * (r1 - r0 @ R)
            D = D / np.vstack(np.linalg.norm(D, axis=1))
            d_orders.append(np.mean(D))

        return t_orders, r_orders, d_orders

    def __get_flctn_not(self, frame):
        """
        flctn_not : non-translational fluctuation
                    (the collective translation is removed)
        """
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no fluctuation in frame {frame}")
        elif frame in self.__flctn_not.keys():
            flctn_not = self.__flctn_not[frame]
        else:
            r0, r1 = self.get_position_pair(frame)
            flctn_not = r1 - r0
            self.__flctn_not.update({frame: flctn_not})
        return flctn_not

    def __get_flctn_noi(self, frame):
        """
        flctn_noi : non-isometric fluctuation
                    (the collective translation & rotaton is removed)
        """
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no fluctuation in frame {frame}")
        elif frame in self.__flctn_noi.keys():
            flctn_noi = self.__flctn_noi[frame]
        else:
            r0, r1 = self.get_position_pair(frame)
            if len(r0) > 3:
                R = self.get_isometry(frame)
                flctn_noi = r1 - r0 @ R
            else:
                flctn_noi = np.empty((0, 3))
            self.__flctn_noi.update({frame: flctn_noi})
        return flctn_noi

    def __get_flctn_nos(self, frame):
        """
        flctn_nos : non-similar fluctuation
                    (the collective translation & rotation & isotropic scaling is removed)
        """
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no fluctuation in frame {frame}")
        elif frame in self.__flctn_nos.keys():
            flctn_nos = self.__flctn_nos[frame]
        else:
            r0, r1 = self.get_position_pair(frame)
            if len(r0) > 3:
                G, R = self.get_similarity(frame)
                flctn_nos = r1 - (r0 @ G) @ R
            else:
                flctn_nos = np.empty((0, 3))
            self.__flctn_nos.update({frame: flctn_nos})
        return flctn_nos

    def get_corr_flctn(self, bins, start, stop=None, transform="T"):
        """
        get the time-average connected correlation function of the fluctuation
        transform:
            T - corr of non-translational fluctuations
            I - corr of non-isometric fluctuations
            S - corr of non-similar fluctuations
        """
        if not stop:
            stop = len(self.movie) - 2

        distances_multi_frame = np.empty(0)
        fluctuations_multi_frame = np.empty(0)

        if transform == "T":
            flctn_func = self.__get_flctn_not
        elif transform == "I":
            flctn_func = self.__get_flctn_noi
        elif transform == "S":
            flctn_func = self.__get_flctn_nos
        else:
            raise ValueError(f"Invalid transformation type: {transform}")

        for frame in range(start, stop+1):
            positions = self.get_position_pair(frame)[0]

            if len(positions) < 4:
                continue

            fluctuations = flctn_func(frame)
            norm = np.mean([f @ f for f in fluctuations])

            if np.isclose(norm, 0):
                continue

            flctn_dimless = fluctuations / np.sqrt(norm)
            distances = pdist(positions)
            flctn_dimless_ij = np.empty(len(distances), dtype=float)

            idx = 0

            for i in range(len(flctn_dimless)):
                for j in range(i+1, len(flctn_dimless)):
                    flctn_dimless_ij[idx] = flctn_dimless[i] @ flctn_dimless[j]
                    idx += 1

            distances_multi_frame = np.concatenate((
                distances_multi_frame,
                distances
            ), axis=0)

            fluctuations_multi_frame = np.concatenate((
                fluctuations_multi_frame,
                flctn_dimless_ij
            ), axis=0)

        return binned_statistic(distances_multi_frame, fluctuations_multi_frame, bins=bins)


class AverageAnalyser():
        """
        Calculate the averaged properties from a movie
            movie :
                a `Movie` instance from package `fish_track`
                    Movie[ f ]             - the positions of all particles in frame f
                    Movie.velocity( f )    - the velocities of all particles in frame f
                    Movie.label( f )       - the labels of all particles in frame f, same label = same identity
                    Movie.trajs[ i ]       - the positions of trajectory i. Here _i_ is the _label_ for these positions
                    Movie.indice_pair( f ) - the paried indices of frame _f_ and frame _f+1_
                                             example:
                                                 p0, p1 = Movie.indice_pair(f)
                                                 Movie[f][p0] & Movie[f+1][p1] correspond to the same particles

        """

    def __init__(self, movie, win_size: int, step_size: int, start=0, end=0):
        """
        :param movie: the movie instance, contains the trajectories, positions and velocitis
                      the movie should be interpolated
        :param win_size: the size of the window (unit frame), in which the average will be calculated
        :param step_size: the average window is moved along the time axis, with the step
        """
        self.movie = movie
        self.start = start
        self.win_size = win_size
        self.step_size = step_size
        if end = 0:
            self.end = movie.max_frame
        else:
            self.end = end

        self.__check_arg(self.win_size, self.start, self.end):

        self.pairs = [(t0, t0 + win_size) for t0 in range(self.start, self.end + 1 - self.win_size, self.step_size)]


    def __check_arg(self, win_size:int, start: int, end: int):
        if self.start > self.end:
            raise ValueError("Starting frame >= ending frame")
        if self.win_size > self.end - self.start + 1:
            raise ValueError("Window size is larger than video length")


    def scan(self, func):
        """
        the data to be averaged is calculated by func(self.movie)
        """
        data = func(self.movie)
        return self.scan_array(data)


    def scan_array(self, array: np.array):
        """
        perform averaging along a 1D numpy array
        """
        if len(array) != self.start - self.end + 1:
            raise ValueError("len(array) != len(movie)")

        result = np.empty(len(self.pairs))

        for i, (t0, t1) in enumerate(self.pairs):
            result[i] = array[t0:t1].mean()

        return result
