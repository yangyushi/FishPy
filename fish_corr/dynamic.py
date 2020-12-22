#!/usr/bin/env python3
import numpy as np
import types
import pickle
from scipy.stats import binned_statistic
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable
from . import utility, static


class Critic():
    """
    Calculate the dynamical order & correlations from a fish movie

    Attributes:
        movie (:obj:`Movie` or :obj:`SimMovie`): a Movie instance where
            positiosn and velocities can be retrieved
        trajs (:obj:`list` of :obj:`fish_track.linking.Trajectory`): a
            list of trajectories
        is_simulation (:obj:`bool`): set the True if the movie is
            :obj:`SimMovie`
        is_pbc (:obj:`bool`): set the True if the particles are in a
            periodic boundary

    Abbrs:
        1. flctn_not: non-translational fluctuation
            (the collective translation is removed)
        2. flctn_noi: non-isometric fluctuation
            (the collective translation & rotaton is removed)
        3. flctn_nos: non-similar fluctuation
            (the collective translation & rotation & isotropic scaling is removed)
    """
    def __init__(self, movie, is_simulation=False, pbc=0):
        self.movie = movie
        self.trajs = movie.trajs
        self.__GR = {}
        self.__R = {}
        self.__velocities = {}
        self.__flctn_not = {}
        self.__flctn_noi = {}
        self.__flctn_nos = {}
        self.is_simulation = is_simulation
        self.pbc = pbc

    def get_position_pair(self, frame, want_flctn=True):
        """
        Retrieve the fluctuation of velocities in ``frame`` and ``frame + 1``

        Args:
            frame (:obj:`int`): the frame number
            want_flctn (:obj:`bool`): if false, the velocities will be returned,
                instead of their fluctuations

        Return:
            :obj:`tuple` ( :obj:`numpy.ndarray`, :obj:`numpy.ndarray` ):
                r0 (shape (n, dim)) ,  r1 (shape (n, dim))
        """
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no position pair in frame {frame}")
        else:
            if self.is_simulation:
                r0 = self.movie[frame].copy()
                v0 = self.movie.velocity(frame).copy()
                if want_flctn:
                    v0 -= np.mean(v0, axis=0)[np.newaxis, :]
                r1 = r0 + v0
            else:
                idx_0, idx_1 = self.movie.indice_pair(frame)
                if len(idx_0) > 0:
                    r0 = self.movie[frame+0][idx_0].copy()
                    r1 = self.movie[frame+1][idx_1].copy()
                    if want_flctn:
                        r0 = r0 - np.mean(r0, axis=0)[np.newaxis, :]
                        r1 = r1 - np.mean(r1, axis=0)[np.newaxis, :]
                else:
                    r0 = np.zeros((0, 3))
                    r1 = np.zeros((0, 3))
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
        Return nan for one frame if the matched pair in that frame is smaller
            than `size_threshold`
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

    def __get_velocities(self, frame):
        if frame > len(self.movie) - 1:
            raise IndexError(f"There is no fluctuation in frame {frame}")
        elif frame in self.__velocities.keys():
            velocities = self.__velocities[frame]
        else:
            r0, r1 = self.get_position_pair(frame, want_flctn=False)
            velocities = r1 - r0
            self.__velocities.update({frame: velocities})
        return velocities

    def __get_flctn_not(self, frame):
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

    def get_corr_flctn(self, bins, start, stop=None, transform="N", get_raw_data=False):
        """
        Get the time-average connected correlation function of the fluctuation

        Args:
            bins (:obj:`int` or :obj:`numpy.ndarray`): the number of bins or
                the bin edges
            start (:obj:`int`): the start frame for the analysis
            stop (:obj:`int` or None): the end frame for the analysis,
                if being `None` the end frame is the last frame
            transform (:obj:`str`): here are possible options

                1. ``T`` - corr of non-translational fluctuations
                2. ``I`` - corr of non-isometric fluctuations
                3. ``S`` - corr of non-similar fluctuations
                4. ``N`` - corr of vanilla fluctuations

            get_raw_data (:obj:`bool`): if True, return the raw data for
                :obj:`scipy.stats.binned_statistic`
                otherwise return the mean value of velocity correlation
        """
        if not stop:
            stop = len(self.movie) - 1

        distances_multi_frame = []
        fluctuations_multi_frame = []

        if transform == "N":
            flctn_func = self.__get_velocities
        elif transform == "T":
            flctn_func = self.__get_flctn_not
        elif transform == "I":
            flctn_func = self.__get_flctn_noi
        elif transform == "S":
            flctn_func = self.__get_flctn_nos
        else:
            raise ValueError(f"Invalid transformation type: {transform}")

        for frame in range(start, stop):
            idx_0, idx_1 = self.movie.indice_pair(frame)

            if len(idx_0) <= 4:
                continue

            positions = self.movie[frame][idx_0]

            fluctuations = flctn_func(frame)
            norm = np.mean([f @ f for f in fluctuations])  # follow attanasi2014pcb

            if np.isclose(norm, 0):
                continue

            flctn_dimless = fluctuations / np.sqrt(norm)

            if self.pbc > 0:
                distances = utility.pdist_pbc(positions, self.pbc)
            else:
                distances = pdist(positions)
            flctn_dimless_ij = np.empty(len(distances), dtype=float)

            idx = 0
            for i in range(len(flctn_dimless)):
                for j in range(i+1, len(flctn_dimless)):
                    flctn_dimless_ij[idx] = flctn_dimless[i] @ flctn_dimless[j]
                    idx += 1

            distances_multi_frame.append(distances)
            fluctuations_multi_frame.append(flctn_dimless_ij)

        distances_multi_frame = np.concatenate(distances_multi_frame, axis=0)
        fluctuations_multi_frame = np.concatenate(fluctuations_multi_frame, axis=0)
        if get_raw_data:
            return distances_multi_frame, fluctuations_multi_frame
        else:
            return binned_statistic(
                distances_multi_frame, fluctuations_multi_frame, bins=bins
            )


class AverageAnalyser():
    """
    Calculate the averaged properties from a movie

    Attributes:
        movie (:obj:`fish_track.linking.Movie`): Movie or SimMovie instance
    """
    def __init__(self, movie, win_size: int, step_size: int, start=0, end=0):
        """
        Args:
            movie (:obj:`fish_track.linking.Movie`):
                A :obj:`fish_track.linking.Movie` instance, being interpolated
            win_size (:obj:`int`):
                the size of the window (unit frame), in which the average will
                be calculated
            step_size (:obj:`int`): the average window is moved along the time
                axis, with the step
        """
        self.movie = movie
        self.start = start
        self.win_size = win_size
        self.step_size = step_size
        self.cache = {}
        if end == 0:
            self.end = movie.max_frame
        else:
            self.end = end

        self.__check_arg()

        self.pairs = [
            (t0, t0 + win_size) for t0 in range(
                self.start,
                self.end - self.win_size,
                self.step_size
            )
        ]
        self.pair_ends = [p[1] - self.start for p in self.pairs]
        self.time = np.array([p[0] + win_size//2 for p in self.pairs], dtype=int)

    def __check_arg(self):
        if self.start > self.end:
            raise ValueError("Starting frame >= ending frame")
        if self.win_size > self.end - self.start + 1:
            raise ValueError("Window size is larger than video length")

    def __scan_positions(self, func) -> np.ndarray:
        """
        The data to be averaged is calculated by func(self.movie)
        The average is tanken between (t0, t1) in self.pairs

        Args:
            func (Callable): a function that operates on a
        """
        result = []
        for i, pair in enumerate(self.pairs):
            res = func(self.movie[pair[0]:pair[1]])
            if type(res) == types.GeneratorType:
                res = np.fromiter(res, dtype=np.float64)
            result.append(res)
        return np.array(result)

    def __scan_velocities(self, func):
        """
        The data to be averaged is calculated by
        The average is tanken between (t0, t1) in self.pairs

        Args:
            func (Callable): func(self.movie.velocities) -> feature

        Return:
            np.ndarray: the average feature value in each average window
        """
        result = []
        for i, pair in enumerate(self.pairs):
            res = func(self.movie.velocity(pair))
            result.append(res)
        return np.array(result)

    def get_trimmed_velocities(self, sample_points, t0, t1):
        """
        Find & Trim trajectories that fits in time range between t0 to t1
            then obtain its velocities

        .. code-block::

                 t0                      t1
                  │    sample_points     │
                  │       ......         │ 1. Valid traj, the size between t0 and
                  │     ─────────▶       │    t1 is larger than sample_points
                  │                      │
                  │       ......         │ 2. Invalid traj, being too short
                  │        ───▶          │
                  │                      │
                ..╬...                   │ 3. Invalid traj, the size between t0
             ─────┼───▶                  │    and t1 is too short, "too_early"
                  │                      │
                  │                   ...╬..   4. Invalid traj, "too_late"
                  │                   ───┼───▶
                  │                      │
                  │       ......         │
             ─────┼──────────────────────┼───▶ 5. Valid traj, but will be trimmed
                  │                      │
            ──────┴──────────────────────┴──────▶ Time

        Args:
            sample_points (:obj:`int`): the length of the obtained ACF
            t0 (:obj:`int`): the start time point of the average window
            t1 (:obj:`int`): the end time point of the average window

        Return:
            :obj:`list` of :obj:`numpy.ndarray`: all the valid velocities
                from the valid and trimmed trajectories
        """
        result = []
        for traj in self.movie.trajs:
            too_early = (traj.t_end - sample_points - 1) < t0
            too_late  = (traj.t_start + sample_points + 1) > t1
            too_short = (len(traj) - 2) < sample_points
            if too_late or too_early or too_short:
                continue
            else:
                offset = max(t0 - traj.t_start, 0)
                stop = min(traj.t_end, t1) - traj.t_start + 1
                if isinstance(
                    traj.velocities, type(None)
                ):  # for experimental data
                    positions = traj.positions[offset : stop]
                    velocities = positions[1:] - positions[:-1]
                else:  # for simulation data
                    velocities = traj.velocities[offset : stop]
                result.append(velocities)
        return result

    def scan_array(self, array):
        """
        Get the average value of an array in all the average windows

        Args:
            array (:obj:`numpy.ndarray`): number of a 1D array, being a feature
                in each frame. The array is assumed to start at frame self.start
        """
        last_pair = 0
        for pe in self.pair_ends:
            if pe <= len(array):
                last_pair += 1
        result = np.empty(last_pair)
        for i, (t0, t1) in enumerate(self.pairs[:last_pair]):
            result[i] = np.nanmean(array[t0:t1])
        return result

    def scan_array_std(self, array: np.ndarray):
        """
        the array is assumed to start at frame self.start
        """
        last_pair = 0
        for pe in self.pair_ends:
            if pe <= len(array):
                last_pair += 1

        result = np.empty(last_pair)
        for i, (t0, t1) in enumerate(self.pairs[:last_pair]):
            result[i] = np.nanstd(array[t0:t1])
        return result

    def scan_nn(self, no_vertices=True):
        if self.win_size >= self.step_size:
            nn_movie = np.fromiter(
                static.get_nn_iter(self.movie, no_vertices=no_vertices),
                dtype=np.float64
            )
            return self.scan_array(nn_movie)
        else:
            return self.__scan_positions(
                lambda x: np.nanmean(
                    np.fromiter(
                        static.get_nn_iter(x, no_vertices=no_vertices),
                        dtype=np.float64
                    ),
                )
            )

    def scan_nn_pbc(self, box):
        if self.win_size >= self.step_size:
            nn_movie = np.array([
                np.nanmean(f) for f in static.get_nn_iter_pbc(self.movie, box)
            ])
            return self.scan_array(nn_movie)
        else:
            return self.__scan_positions(
                lambda x: np.nanmean(
                    list(static.get_nn_iter_pbc(x, box=box))
                ),
            )

    def scan_nn_std(self, no_vertices=True):
        if self.win_size >= self.step_size:
            nn_movie = np.fromiter(
                static.get_nn_iter(self.movie, no_vertices=no_vertices),
                dtype=np.float64
            )
            return self.scan_array_std(nn_movie)
        else:
            return self.__scan_positions(
                lambda x: np.nanstd(
                    np.fromiter(
                        static.get_nn_iter(x, no_vertices=no_vertices),
                        dtype=np.float64
                    )
                ),
            )

    def scan_nn_pbc_std(self, box):
        if self.win_size >= self.step_size:
            nn_movie = np.array([
                np.nanmean(f) for f in static.get_nn_iter_pbc(self.movie, box)
            ])
            return self.scan_array_std(nn_movie)
        else:
            return self.__scan_positions(
                lambda x: np.nanstd(static.get_nn_iter_pbc(x, box=box)),
            )

    def scan_speed(self, min_number=0):
        """
        Args:
            min_number (:obj:`int`): only take frame into consideration if
                len(velocity) > min_number in this frame
        """
        return self.__scan_velocities(
            lambda x: utility.get_mean_spd(
                x, frame_number=self.win_size,
                min_number=min_number
            )
        )

    def scan_speed_std(self, min_number=0):
        """
        Args:
            min_number (:obj:`int`): only take frame into consideration if
                len(velocity) > min_number in this frame
        """
        return self.__scan_velocities(
            lambda x: utility.get_std_spd(
                x, frame_number=self.win_size,
                min_number=min_number
            )
        )

    def scan_gr(self, tank, bins, number):
        """
        Args:
            tank (:obj:`fish_corr.static.Tank`): a instance of :obj:`Tank`
                to perform random sampling
            bins (:obj:`numpy.ndarray`): the bins for :func:`numpy.histogram`
            number (:obj:`int`): number of (posiible) particles per frame

        Return:
            :obj:`numpy.ndarray`: The radial distribution function
        """
        return self.__scan_positions(
            lambda x: utility.get_vanilla_gr(
                x, tank=tank, bins=bins, random_size=number * self.win_size
            )
        )

    def scan_biased_gr(self, bins, space_bin_number=50, **kwargs):
        """
        Args:
            bins (:obj:`numpy.ndarray`): the bins for :func:`numpy.histogram`
            space_bin_number (:obj:`int`): number of bins to devide space to
                generated biased random gas

        Return:
            :obj:`numpy.ndarray`: The radial distribution function assuming
                biased density distribution
        """
        positions = []
        for frame in self.movie:
            if len(frame) > 0:
                positions.append(frame)
        positions = np.vstack(positions)
        return self.__scan_positions(
            lambda x: utility.get_biased_gr(
                frames=x, positions=positions,
                bins=bins, space_bin_number=space_bin_number
            )
        )

    def scan_biased_attraction(self, bins, space_bin_number, **kwargs):
        """
        Args:
            bins (:obj:`numpy.ndarray`): the bins for :func:`numpy.histogram`
            space_bin_number (:obj:`int`): number of bins to devide space to
                generated biased random gas

        Return:
            :obj:`numpy.ndarray`: the effective attraction in different average windows
        """
        biased_rdfs = self.scan_biased_gr(bins, space_bin_number)
        attractions = -np.log(np.max(biased_rdfs, axis=1))
        return attractions

    def scan_biased_attraction_err(self, bins, space_bin_number, repeat, **kwargs):
        """
        Using the bootstrap method to

        Args:
            bins (:obj:`numpy.ndarray`): the bins for :func:`numpy.histogram`
            space_bin_number (:obj:`int`): number of bins to devide space to
                generated biased random gas

        Return:
            :obj:`numpy.ndarray`: the strandard error of the effective attraction
                in different average windows
        """
        positions = []
        for frame in self.movie:
            if len(frame) > 0:
                positions.append(frame)
        positions = np.vstack(positions)

        result = []
        for _ in range(repeat):
            rdfs = self.__scan_positions(
                lambda x: utility.get_biased_gr_randomly(
                    x, positions=positions,
                    bins=bins, space_bin_number=space_bin_number
                )
            )
            attractions = -np.log(np.max(rdfs, axis=1))
            result.append(attractions)
        error = np.std(result, axis=0)
        return error

    def scan_vicsek_order(self, min_number=0):
        """
        Args:
            min_number (:obj:`int`): only take frame into consideration
                                     if ``len(velocity) > min_number`` in this frame
        Return:
            :obj:`numpy.ndarray`: The moving-average of the Vicsek order parameter
        """
        if self.win_size >= self.step_size:
            velocities = self.movie.velocity((self.start, self.end))
            vicsek_movie = utility.get_vicsek_order(
                velocities, min_number=min_number
            )
            return self.scan_array(vicsek_movie)
        else:
            return self.__scan_velocities(
                lambda x: np.nanmean(utility.get_vicsek_order(
                    x, min_number=min_number
                ))
            )

    def scan_vicsek_order_std(self, min_number=0):
        """
        Args:
            min_number (:obj:`int`): only take frame into consideration
                                     if ``len(velocity) > min_number`` in this frame
        Return:
            :obj:`numpy.ndarray`: The moving-average of the Vicsek order parameter
        """
        if self.win_size >= self.step_size:
            velocities = self.movie.velocity((self.start, self.end))
            vicsek_movie = utility.get_vicsek_order(
                velocities, min_number=min_number
            )
            return self.scan_array_std(vicsek_movie)
        else:
            return self.__scan_velocities(
                lambda x: np.nanstd(utility.get_vicsek_order(
                    x, min_number=min_number
                ))
            )

    def scan_orientation_acf(self, sample_points: int):
        r"""
        Calculate the averaged rotational relaxation time for the movie

        Args:
            sample_points (:obj:`int`): the maximum lag time (:math:`\tau`)
                in the ACF calculation (:func:`get_acf`)

        Return:
            the acf functions in different time windows
        """
        result = []
        for i, (t0, t1) in enumerate(self.pairs):
            acfs = []
            for velocities in self.get_trimmed_velocities(sample_points, t0, t1):
                norms = np.linalg.norm(velocities, axis=1)
                norms[np.isclose(norms, 0)] = np.nan
                orientations = velocities / norms[:, np.newaxis]
                acf = utility.get_acf(orientations, size=sample_points)
                acfs.append(acf)
            if len(acfs) == 0:
                null_acf = np.zeros((1, sample_points))
                null_acf[:] = np.nan
                result.append(null_acf)
            else:
                result.append(np.array(acfs))
        return result

    def scan_rotation(self, sample_points):
        """
        Scan the average relaxation time of the orientation
        """
        acfs_mw = self.scan_orientation_acf(sample_points)  # mw = multiple windows
        result = np.empty(len(acfs_mw))
        for i, acfs in enumerate(acfs_mw):
            acf = np.nanmean(acfs, axis=0)
            if np.nan in acf:
                result[i] = np.nan
            else:
                result[i] = utility.fit_acf_exp(acf)
        return result

    def scan_rotation_err(self, sample_points, repeat=10):
        """
        Get the standard error of the rotational relaxation time (tau) using
            bootstrap method
        """
        acfs_mw = self.scan_orientation_acf(sample_points)  # mw = multiple windows
        result = np.empty(len(acfs_mw))
        for i, acfs in enumerate(acfs_mw):
            if np.nan in acfs:
                result[i] = np.nan
            else:
                tau_vals = np.empty(repeat)
                # bootstrap method to infer the error
                for j in range(repeat):
                    random_indices = np.random.randint(0, len(acfs), len(acfs))
                    acf = np.mean(acfs[random_indices], axis=0)
                    tau_vals[j] = utility.fit_acf_exp(acf)
                result[i] = np.nanstd(tau_vals)
        return result

    def scan_number(self):
        numbers = np.array([len(frame) for frame in self.movie])
        return self.scan_array(numbers)

    def decorrelated_scan_2(self, f1, f2):
        """
        Decorrelated version of :any:`scan_array()`
            the time averaged signals were averaged by randomly selected data
            points to reduce possible error-induced correlation

        Args:
            f1 (:obj:`function`): the function to generate signal,
                :code:`f1(self) -> signal_1`
            f2 (:obj:`function`): the function to generate signal,
                :code:`f2(self) -> signal_2`

        Return:
            :obj:`tuple` ( :obj:`numpy.ndarray` , :obj:`numpy.ndarray` ):
                the time averaged signals
        """
        x1 = f1(self)
        x2 = f2(self)
        assert len(x1) == len(x2),\
            f"can't decorrelate signals with different sizes, ({len(x1)} != {len(x2)})"
        last_pair = 0
        for pe in self.pair_ends:
            if pe <= len(x1):
                last_pair += 1
        result_1 = np.empty(last_pair)
        result_2 = np.empty(last_pair)
        for i, (t0, t1) in enumerate(self.pairs[:last_pair]):
            indices = np.arange(t0, t1, 1)
            np.random.shuffle(indices)
            i1 = indices[:self.win_size//2]
            i2 = indices[self.win_size//2:]
            result_1[i] = np.nanmean(x1[i1])
            result_2[i] = np.nanmean(x2[i2])
        return result_1, result_2

    def decorrelated_scan_3(self, f1, f2, f3):
        """
        Decorrelated version of :any:`scan_array()`
            the time averaged signals were averaged by randomly selected data points
            to reduce possible error-induced correlation

        Args:
            f1 (:obj:`function`): the function to generate signal,
                :code:`f1(self) -> signal_1`
            f2 (:obj:`function`): the function to generate signal,
                :code:`f2(self) -> signal_2`
            f3 (:obj:`function`): the function to generate signal,
                :code:`f3(self) -> signal_3`

        Return:
            :obj:`tuple` of :obj:`numpy.ndarray`: the time averaged signals
        """
        x1, x2, x3 = f1(self), f2(self), f3(self)
        assert (len(x1) == len(x2)) and (len(x1) == len(x3)),\
        f"Invalid signal sizes, ({len(x1)}, {len(x2)}, {len(x3)})"
        last_pair = 0
        for pe in self.pair_ends:
            if pe <= len(x1):
                last_pair += 1
        result_1 = np.empty(last_pair)
        result_2 = np.empty(last_pair)
        result_3 = np.empty(last_pair)
        for i, (t0, t1) in enumerate(self.pairs[:last_pair]):
            indices = np.arange(t0, t1, 1)
            np.random.shuffle(indices)
            i1 = indices[ : self.win_size//3]
            i2 = indices[self.win_size//3 : 2 * self.win_size//3]
            i3 = indices[2 * self.win_size//3 : ]
            result_1[i] = np.nanmean(x1[i1])
            result_2[i] = np.nanmean(x2[i2])
            result_3[i] = np.nanmean(x3[i3])
        return result_1, result_2, result_3


def get_nn_movie(analyser):
    """
    get the average nearest neighbour distance of all frames from a analyser
    this is intended to be used for AverageAnalyser.decorrelated_scan_3()
    or AverageAnalyser.decorrelated_scan_3()

    Args:
        analyser (AverageAnalyser): instance of the AverageAnalyser

    Return:
        (:obj:`numpy.ndarray`): the average nn-distance in all frames
    """
    frames = analyser.movie[analyser.start : analyser.end]
    nn_iter = static.get_nn_iter(frames, no_vertices=True)
    return np.fromiter(nn_iter, dtype=float)


def get_spd_movie(analyser):
    """
    get the average speed of all frames from a analyser
        this is intended to be used for AverageAnalyser.decorrelated_scan_3()
        or AverageAnalyser.decorrelated_scan_3()

    Args:
        analyser (AverageAnalyser): instance of the AverageAnalyser

    Return:
        (:obj:`numpy.ndarray`): the average speed in all frames
    """
    spd_movie = np.empty(analyser.end - analyser.start)
    for i, f in enumerate(range(analyser.start, analyser.end)):
        velocity = analyser.movie.velocity(f)
        if len(velocity) > 0:
            spd = np.linalg.norm(velocity, axis=1)
            if np.logical_not(np.isnan(spd)).sum() > 0:
                spd_movie[i] = np.nanmean(spd)
            else:
                spd_movie[i] = np.nan
    return spd_movie


def get_order_movie(analyser):
    """
    get the dynamical order of all frames from a analyser
    this is intended to be used for AverageAnalyser.decorrelated_scan_3()
    or AverageAnalyser.decorrelated_scan_3()

    Args:
        analyser (AverageAnalyser): instance of the AverageAnalyser

    Return:
        (:obj:`numpy.ndarray`): the dynamical order (polarisation) in all frames
    """
    velocities = analyser.movie.velocity((analyser.start, analyser.end))
    vicsek_movie = utility.get_vicsek_order(velocities, min_number=2)
    return vicsek_movie


