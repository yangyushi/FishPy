#!/usr/bin/env python3
import cv2
import pickle
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
import matplotlib.pyplot as plt
from typing import List, Tuple
from .cstereo import refractive_project


def draw(img, axes):
    origin = tuple(axes[0].ravel().astype(int))
    img = cv2.line(
            img, origin, tuple(axes[1].ravel()),
            (255, 120, 100), 5, lineType=cv2.LINE_AA
            )
    img = cv2.line(
            img, origin, tuple(axes[2].ravel()),
            (100, 255, 120), 5, lineType=cv2.LINE_AA
            )
    img = cv2.line(
            img, origin, tuple(axes[3].ravel()),
            (100, 120, 255), 5, lineType=cv2.LINE_AA
            )
    return img


def find_pairs(arr_1, arr_2):
    assert len(arr_1.ravel()) == len(set(arr_1.ravel()))
    assert len(arr_2.ravel()) == len(set(arr_2.ravel()))
    assert set(arr_1.ravel()) == set(arr_2.ravel())
    pairs = [[], []]
    for i in range(arr_2.shape[0]):
        for j in range(arr_2.shape[1]):
            index_1 = np.squeeze(np.where(arr_1 == arr_2[i, j]))
            pairs[0].append(index_1)  # arr_1
            pairs[1].append((i, j))  # arr_2
    return np.array(pairs)


def get_points_from_order(corner_number, order='x123'):
    """
    the expected calibration image is

    .. code-block::

        ┌───────┬───────┬───────┐
        │  ╲ ╱  │◜◜◜◜◜◜◜│   ╮   │
        │   ╳   │◜◜◜◜◜◜◜│   │   │
        │  ╱ ╲  │◜◜◜◜◜◜◜│   ┴   │
        ├───────┼───────┼───────┤
        │◜◜◜◜◜◜◜│       │◜◜◜◜◜◜◜│
        │◜◜◜◜◜◜◜│       │◜◜◜◜◜◜◜│
        │◜◜◜◜◜◜◜│       │◜◜◜◜◜◜◜│
        ├───────┼───────┼───────┤
        │  ╶─╮  │◜◜◜◜◜◜◜│   ──┐ │
        │  ╭─╯  │◜◜◜◜◜◜◜│   ╶─┤ │
        │  ╰──  │◜◜◜◜◜◜◜│   ──┘ │
        └───────┴───────┴───────┘

    and the corresponding order is x123 (row, colume)
    """
    obj_points = []
    standard_order, standard = 'x123', np.arange(4).reshape(2, 2)
    reality = np.array(
            [standard_order.index(letter) for letter in order], dtype=int
            ).reshape(2, 2)

    pairs = find_pairs(standard, reality)
    pairs = 2 * pairs - 1

    transformations = []
    for angle in [0, 0.5 * np.pi, np.pi, 1.5 * np.pi]:
        for inv in [1, -1]:
            trans = np.zeros((2, 2), dtype=np.float32)
            trans[0, 0] = np.cos(angle) * inv
            trans[0, 1] = -np.sin(angle)
            trans[1, 0] = np.sin(angle) * inv
            trans[1, 1] = np.cos(angle)
            transformations.append(trans)

    for t in transformations:
        err = np.zeros((2, 2))
        for p1, p2 in zip(*pairs):
            err += np.abs(t @ p1 - p2)  # t @ std -> real
        if np.allclose(err, 0):
            break

    if not np.allclose(err, 0):
        raise RuntimeError("Impossible order")

    std = []
    centre = np.ones(2, dtype=float) * (np.array(corner_number)[::-1] - 1) / 2
    for c1 in range(corner_number[1]):
        for c2 in range(corner_number[0]):
            p = np.array((c1, c2), dtype=centre.dtype)
            std.append(p.copy())
            p -= centre
            p = t @ p  # std -> real
            obj_points.append(np.hstack([p, 0]))
    std = np.array(std)
    count = 0
    obj_points = np.array(obj_points, dtype=np.float32)
    obj_points -= obj_points.min(0)

    if 0:
        for s, r in zip(std, obj_points):
            c = np.random.random(3)
            plt.scatter(*s, color=c, marker='.', s=200)
            plt.scatter(*r[:2], color=c, facecolor='none', s=100)
            plt.quiver(*s, *(r[:2]-s), color=c)
            count += 1
        plt.show()
    return obj_points


def plot_cameras(axis, cameras, water_level=0, depth=400):
    origins = []
    camera_size = 100
    focal_len = 2
    ray_length = 400
    camera_segments = [
            np.array(([0, 0, 0], [1, -1, focal_len])) * camera_size,
            np.array(([0, 0, 0], [-1, 1, focal_len])) * camera_size,
            np.array(([0, 0, 0], [-1, -1, focal_len])) * camera_size,
            np.array(([0, 0, 0], [1, 1, focal_len])) * camera_size,
            np.array(([1, 1, focal_len], [1, -1, focal_len])) * camera_size,
            np.array(([1, -1, focal_len], [-1, -1, focal_len])) * camera_size,
            np.array(([-1, -1, focal_len], [-1, 1, focal_len])) * camera_size,
            np.array(([-1, 1, focal_len], [1, 1, focal_len])) * camera_size
            ]
    for cam in cameras:
        origin = -cam.r.T @ cam.t
        orient = cam.r.T @ np.array([0, 0, 1])
        origins.append(origin)
        for seg in camera_segments:
            to_plot = np.array([cam.r.T @ p + origin for p in seg])
            axis.plot(*to_plot.T, color='b')
        axis.scatter(*origin, color='w', edgecolor='k')
        axis.quiver(*origin, *orient * ray_length, color='k')

    xlim, ylim, zlim = np.array(origins).T
    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    x = np.linspace(mid_x - 2e3, mid_x + 2e3, 11, endpoint=True)
    y = np.linspace(mid_y - 2e3, mid_y + 2e3, 11, endpoint=True)
    x, y = np.meshgrid(x, y)
    axis.plot_surface(x, y, np.ones(x.shape) * water_level, alpha=0.3)
    axis.set_zlim(-depth, 2000)
    return axis


def get_reproject_error(
        points_2d, points_3d,
        rvec, tvec, distort, camera_matrix
        ):
    projected, _ = cv2.projectPoints(
            points_3d, rvec, tvec,
            cameraMatrix=camera_matrix,
            distCoeffs=distort
            )
    shift = np.squeeze(projected) - np.squeeze(points_2d)
    return np.linalg.norm(shift, axis=1).mean()


class Camera():
    """
    This class holds intrinsic/extrinsic parameters of a camera, with some handy functions

    Attributes:
        self.r: rotation matrix of the camera, shape (3, 3)
        self.ext: extrinsic parameters, # R, t --> [R|t]
        self.p: projection matrix int @ ext
        self.o: origin of the camera, shape (3, 1)
    """
    def __init__(self):
        # essential parameters
        self.rotation = R.from_rotvec(np.zeros(3))
        self.distortion = np.zeros(5)
        self.t = np.zeros(3)
        self.k = np.identity(3)
        # updated parameters
        self.r = np.zeros((3, 3))
        self.ext = np.zeros((3, 4))
        self.o = np.zeros(3)
        self.p = np.zeros((3, 4))
        self.calibration_files = []
        self.update()

    def __str__(self):
        def ff(x):
            return np.array2string(
                    x, precision=4, floatmode='fixed', separator='\t'
                    )
        info_1 = f'Camera instance @{id(self):x}\n'
        info_2 = f'Intrinsic Matrix is \n {ff(self.k)}\n'
        info_3 = f'Extrinsic Matrix is \n {ff(self.ext)}\n'
        info_4 = f'Distortion is \n {ff(self.distortion)}\n'
        return info_1 + info_2 + info_3 + info_4

    def update(self):
        try:
            self.r = self.rotation.as_matrix()  # scipy > 1.4.0
        except AttributeError:
            self.r = self.rotation.as_dcm()  # scipy < 1.4.0
        self.ext = np.hstack([self.r, np.vstack(self.t)])
        self.p = np.dot(self.k, self.ext)
        self.o = np.vstack(-self.r.T @ self.t)

    def read_calibration(self, mat_file: str):
        """
        Read calibration result from TOOLBOX_calib
        The calibration result is generated by following Matlab script:

        .. highlight:: matlab
        .. code-block:: matlab

            save(filename, 'fc', 'cc', 'Tc_ext', 'Rc_ext');
        """
        calib_result = loadmat(mat_file)
        self.k[0][0] = calib_result['fc'][0, 0]
        self.k[1][1] = calib_result['fc'][1, 0]
        self.k[0][2] = calib_result['cc'][0, 0]
        self.k[1][2] = calib_result['cc'][1, 0]
        if 'kc' in calib_result.keys():
            self.distortion = np.array(calib_result['kc']).ravel()
        if 'Tc_ext' in calib_result.keys():
            self.t = calib_result['Tc_ext'][:, 0]
        if 'Rc_ext' in calib_result.keys():
            self.rotation = R.from_dcm(calib_result['Rc_ext'])
        self.update()

    def read_int(self, pkl_file: str):
        with open(pkl_file, 'rb') as f:
            cam = pickle.load(f)
        self.k = cam.k
        self.distortion = cam.distortion
        self.update()

    def project(self, position: np.array):
        """
        Project a 3D position onto the image plane

        Args:
            position (np.ndarray): a collection of 3D poitns, shape (n, 3)
        """
        if (position.ndim == 1) and (position.shape[0] == 3):
            position = position[np.newaxis, :] # try to convert (3,) to (1, 3)
        assert position.shape[1] == 3, "Please input an (n, 3) array"
        uv, _ = cv2.projectPoints(
                objectPoints=np.vstack(position.T).T,
                rvec=self.rotation.as_rotvec(),
                tvec=self.t,
                cameraMatrix=self.k,
                distCoeffs=self.distortion
        )
        return np.squeeze(uv)

    def project_refractive(self, positions):
        """
        Project the 3D points under water
        the normal water-air interface is assumed to be (0, 0, 1)

        Args:
            positions (:obj:`numpy.ndarray`): a collection of 3d points, shape (n, 3)

        Return:
            :obj:`numpy.ndarray`: the projected 2D locations, shape (n, 2)
        """
        coord_2d_nodist = refractive_project(positions, self.p, self.o)
        coord_2d = self.redistort_points(coord_2d_nodist.T)
        return coord_2d.T

    def undistort(self, point: np.array, want_uv=False):
        """
        undistort point in an image, coordinate is (u, v)

        Args:
            point (np.ndarray): the points to be undistorted
            want_uv (bool):

        Return:
            np.ndarray: undistorted points, being xy, not uv (camera.K @ xy = uv)
        """
        new_point = point.astype(np.float64)
        new_point = np.expand_dims(new_point, 0)
        new_point = np.expand_dims(new_point, 0)
        if want_uv:
            undistorted = cv2.undistortPoints(
                    src=new_point,
                    cameraMatrix=self.k,
                    distCoeffs=self.distortion,
                    P=self.k,
            )
        else:
            undistorted = cv2.undistortPoints(
                    src=new_point,
                    cameraMatrix=self.k,
                    distCoeffs=self.distortion,
                    )
        return np.squeeze(undistorted)

    def undistort_image(self, image: np.array):
        """
        Args:
            image (np.ndarray): an image taken by the camera

        Return:
            np.ndarray: a undistorted 2D image
        """
        return cv2.undistort(image, self.k, self.distortion)

    def undistort_points(self, points: np.array, want_uv=True):
        """
        Undistort many points in an image, coordinate is (u, v), NOT (x, y)

        Return:
            np.ndarray: undistorted version of (x', y') or (u', v'), shape (n, 2)

        .. code-block::

            x' * fx + cx -> u'
        """
        new_points = points.astype(np.float64)
        if new_points.shape[0] == 0:
            return np.empty((0, 2))
        new_points = np.expand_dims(new_points, 1)  # (n, 2) --> (n, 1, 2)
        if want_uv:
            undistorted = cv2.undistortPoints(
                    src=new_points,
                    cameraMatrix=self.k,
                    distCoeffs=self.distortion,
                    P=self.k,
            )
        else:
            undistorted = cv2.undistortPoints(
                    src=new_points,
                    cameraMatrix=self.k,
                    distCoeffs=self.distortion,
            )
        return np.squeeze(undistorted)

    def redistort_points(self, points: np.array):
        """
        Args:
            points (np.ndarray): undistorted image coordinates (u, v) in pixels, shape (2, n)

        Return:
            np.ndarray: the distorted points
        """
        k1, k2, p1, p2, k3 = self.distortion
        fx, fy = self.k[0, 0], self.k[1, 1]
        cx, cy = self.k[:2, 2]
        xy = points - np.vstack((cx, cy))  # shift to centre
        x, y = xy / np.vstack((fx, fy))  # mm --> unitless
        r2 = x ** 2 + y ** 2
        k = 1 + k1 * r2 + k2 * r2**2 + k3 * r2 ** 3
        dist_x = x * k + 2 * p1 * x*y + p2 * (r2 + 2 * x**2)
        dist_y = y * k + p1 * (r2 + 2 * y**2) + 2 * p2 * x*y
        dist_x = dist_x * fx + cx
        dist_y = dist_y * fy + cy
        return np.vstack((dist_x, dist_y))

    def calibrate_int(
            self, int_images: list, grid_size: float,
            corner_number=(6, 6), win_size=(5, 5), show=True
            ):
        """
        update INTERINSIC camera matrix using opencv's chessboard detector
        the distortion coefficients are also being detected
        the corner number should be in the format of (row, column)
        """
        # termination criteria
        criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1
                )

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        detected_indices = []  # indice of images whoes corners were detected

        for i, fname in enumerate(int_images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray, corner_number,
                flags=sum((
                    cv2.CALIB_CB_FAST_CHECK,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    ))
            )
            if ret:
                obj_points_single = get_points_from_order( corner_number, order='x123') * grid_size

                obj_points.append(obj_points_single)

                corners_refined = cv2.cornerSubPix( gray, corners, win_size, (-1, -1), criteria)

                img_points.append(corners_refined)

                img = cv2.drawChessboardCorners( img, corner_number, corners_refined, ret)

                img = cv2.resize(img, (800, 600))
                detected_indices.append(i)
                if show:
                    cv2.imshow('img', img)
                    cv2.waitKey(100)
            else:
                print(f"corner detection failed: {fname}")

        obj_points = np.array(obj_points)
        img_points = np.array(img_points)

        # this initial guess is for basler AC2040 120um camera
        # with a 6mm focal length lens
        camera_matrix = np.array([
            [1739.13, 0, 1024],
            [0, 1739.13, 768],
            [0, 0, 1]
            ])

        ret, camera_matrix, distortion, rvecs, tvecs, std_i, std_e, pve = cv2.calibrateCameraExtended(
                objectPoints=obj_points, imagePoints=img_points,
                imageSize=gray.shape,
                cameraMatrix = camera_matrix,
                distCoeffs = np.zeros(5),
                flags=sum((
                    cv2.CALIB_USE_INTRINSIC_GUESS,
                    #cv2.CALIB_FIX_ASPECT_RATIO,
                    #cv2.CALIB_FIX_PRINCIPAL_POINT,
                    #cv2.CALIB_ZERO_TANGENT_DIST,
                    #cv2.CALIB_FIX_K1,
                    #cv2.CALIB_FIX_K2,
                    cv2.CALIB_FIX_K3,
                    #cv2.CALIB_RATIONAL_MODEL
                    )),
        )

        for i, fname in enumerate(int_images):
            if i in detected_indices:
                j = detected_indices.index(i)
                err = get_reproject_error(
                        img_points[j], obj_points[j],
                        rvecs[j], tvecs[j], distortion, camera_matrix
                        )
                if show:
                    print(f"reproject error for {fname:<12} is {err:.4f}")

        self.k = camera_matrix
        if show:
            print("==== Intrinsic Matrix ====")
            print(self.k)
            print("==== Dist coefficients ====\n", distortion)
        self.distortion = distortion
        self.update()

    def calibrate_ext(self, ext_image: str, grid_size: float, order='x123', corner_number=(6, 6), win_size=(5, 5), show=True):
        """
        update EXTRINSIC camera matrix using opencv's chessboard detector
        the distortion coefficients are also being detected
        the corner number should be in the format of (row, column)
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

        for_plot = []
        img = cv2.imread(ext_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, corner_number,
                flags=sum((
                    cv2.CALIB_CB_FAST_CHECK,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    )))
        if ret == True:
            obj_points = get_points_from_order(corner_number, order=order) * grid_size
            img_points = cv2.cornerSubPix(gray, corners, win_size, (-1,-1), criteria)
            for_plot += [corner_number, img_points, ret]
        else:
            raise RuntimeError("Corner detection failed!")

        ret, rvec, tvec = cv2.solvePnP(
                objectPoints=obj_points,
                imagePoints=img_points,
                cameraMatrix=self.k,
                distCoeffs=self.distortion,
        )

        err = get_reproject_error(
                img_points, obj_points,
                rvec, tvec, self.distortion, self.k
                )

        self.rotation = R.from_rotvec(rvec.ravel())
        self.t = np.ravel(tvec)
        self.update()

        print(f"==== reproject error for Extrinsic is {err:.4f} ====")
        if show:
            length = 100
            axes = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
            axes_img, _ = cv2.projectPoints(axes, rvec, tvec, self.k, self.distortion)
            img = cv2.imread(ext_image)
            img = cv2.drawChessboardCorners(img, *for_plot)
            img = draw(img, axes_img)
            img = cv2.resize(img, (800, 600))
            cv2.imshow('img', img)
            cv2.waitKey(5000)

    def calibrate(self, int_images: list, ext_image: str, grid_size: float, order='x123', corner_number=(6, 6), win_size=(5, 5), show=True):
        """
        update intrinsic and extrinsic camera matrix using opencv's chessboard detector
        the distortion coefficients are also being detected
        the corner number should be in the format of (row, column)
        """
        self.calibrate_int(int_images, grid_size, corner_number, win_size, show)
        self.calibrate_ext(ext_image, grid_size, order, corner_number, win_size, show)

    def save(self, fname: str):
        """
        Save instance as a python binary object
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def zip_essential(self):
        """
        pack all essential parameters into a dict which can be dumped
            as a json file
        """
        try:
            rot_mat = self.rotation.as_matrix()
        except AttributeError:
            rot_mat = self.rotation.as_dcm()
        data = {
            "k": self.k.tolist(),
            "r": rot_mat.tolist(),
            "t": self.t.tolist(),
            "distortion": self.distortion.tolist()
        }
        return data

    def unzip_essential(self, data):
        """
        Unpack essential parameters from a dict and load them as attributes
        """
        self.r = np.array(data['r'])
        try:
            self.rotation = R.from_matrix(self.r)
        except AttributeError:
            self.rotation = R.from_dcm(self.r)
        self.k = np.array(data['k'])
        self.t = np.array(data['t'])
        self.distortion = np.array(data['distortion'])
        self.update()


    def save_json(self, fname):
        """
        Dump essential parameters to a json file
        """
        if fname[-5:] != ".json":
            fname += ".json"
        data = self.zip_essential()
        with open(fname, 'w') as f:
            json.dump(data, f, indent=" " * 4)

    def load_json(self, fname):
        """
        Recover essential parameters from a json file
        """
        with open(fname, 'r') as f:
            data = json.load(f)
        self.unzip_essential(data)

def calib_mult_ext(
    cam_1, cam_2, cam_3, images_v1, images_v2, images_v3,
    orders_v1, orders_v2, orders_v3, grid_size, corner_number,
    win_size=(10, 10), debug=False
):
    """
    Do extrinsic calibrations multiple times, calculate average *relative displacement & angle*
    Then use the **last** calibration image as world coordinate

    Args:
        cam_1 (Camera): The internal parameters should be correct
        cam_2 (Camera): The internal parameters should be correct
        cam_3 (Camera): The internal parameters should be correct
        images_v1 (list): calibrations image filenames for camera 1
        images_v2 (list): calibrations image filenames for camera 2
        images_v3 (list): calibrations image filenames for camera 3
        orders_v1 (list): The order of 'x123' or '321x' for each calibration imags for camera 1
        orders_v2 (list): The order of 'x123' or '321x' for each calibration imags for camera 2
        orders_v3 (list): The order of 'x123' or '321x' for each calibration imags for camera 3
        grid_size (float): size of a single grid in chessboard, only support square grids
        corner_number (tuple): (horizontal, vertical), specifying the number of corners in each direction
                               corner_number = grid_number - 1
        win_size (tuple): Search parameter in the opencv

    Equations (@ is dot product)

    .. code-block::

        x1 = r1  @ xw + t1   (world to camera 1, r1 & t1 obtained from Camera.calibrate_ext)
        x2 = r2  @ xw + t2   (world to camera 2, r2 & t2 obtained from Camera.calibrate_ext)
        x2 = r12 @ x1 + t12  (camera 1 to camera 2)
        --> r12 = r2 @ r1'
        --> t12 = t2 - r2 @ r1' @ t1
        --> t2  = t12 + r12 @ t1
        --> r2  = r12 @ r1
    """
    cam_1_rotations, cam_1_translations = [], []
    cam_2_rotations, cam_2_translations = [], []
    cam_3_rotations, cam_3_translations = [], []

    for fn, order in zip(images_v1, orders_v1):
        cam_1.calibrate_ext(fn, grid_size, order, corner_number, show=False)
        cam_1_rotations.append(cam_1.r)
        cam_1_translations.append(cam_1.t)

    for fn, order in zip(images_v2, orders_v2):
        cam_2.calibrate_ext(fn, grid_size, order, corner_number, show=False)
        cam_2_rotations.append(cam_2.r)
        cam_2_translations.append(cam_2.t)

    for fn, order in zip(images_v3, orders_v3):
        cam_3.calibrate_ext(fn, grid_size, order, corner_number, show=False)
        cam_3_rotations.append(cam_3.r)
        cam_3_translations.append(cam_3.t)

    rotations_12, translations_12 = [], []
    rotations_13, translations_13 = [], []

    for r1, t1, r2, t2 in zip(cam_1_rotations, cam_1_translations, cam_2_rotations, cam_2_translations):
        rotations_12.append(r2 @ r1.T)
        translations_12.append(t2 - r2 @ r1.T @ t1)

    for r1, t1, r3, t3 in zip(cam_1_rotations, cam_1_translations, cam_3_rotations, cam_3_translations):
        rotations_13.append(r3 @ r1.T)
        translations_13.append(t3 - r3 @ r1.T @ t1)

    r12 = np.mean(rotations_12, axis=0)
    r12_std = np.std(rotations_12, axis=0)
    t12 = np.mean(translations_12, axis=0)
    t12_std = np.std(translations_12, axis=0)
    r13 = np.mean(rotations_13, axis=0)
    r13_std = np.std(rotations_13, axis=0)
    t13 = np.mean(translations_13, axis=0)
    t13_std = np.std(translations_13, axis=0)

    r12_xyz = R.from_dcm(r12).as_rotvec() / np.pi * 180
    r12_xyz_std = R.from_dcm(r12_std).as_rotvec() / np.pi * 180

    print(f'Rotation between view#1 and view#2 is {[f"{m:.4f} ± {s:.4f}" for m, s in zip(r12_xyz, r12_xyz_std)]}')
    print(f'Translation between view#1 and view#2 is {[f"{m:.4f} ± {s:.4f}" for m, s in zip(t12, t12_std)]}')

    r13_xyz = R.from_dcm(r13).as_rotvec() / np.pi * 180
    r13_xyz_std = R.from_dcm(r13_std).as_rotvec() / np.pi * 180

    print(f'Rotation between view#1 and view#3 is {[f"{m:.4f} ± {s:.4f}" for m, s in zip(r13_xyz, r13_xyz_std)]}')
    print(f'Translation between view#1 and view#3 is {[f"{m:.4f} ± {s:.4f}" for m, s in zip(t13, t13_std)]}')

    if debug:
        with np.printoptions(precision=2, suppress=True):
            for i, (ti, tj) in enumerate(zip(translations_12, translations_13)):
                print(f"Image #{i+1}, relative shift 12: {ti}; 13: {tj}")

    cam_2.rotation = R.from_dcm(r12 @ cam_1.r)
    cam_2.t = r12 @ cam_1.t + t12
    cam_2.update()

    cam_3.rotation = R.from_dcm(r13 @ cam_1.r)
    cam_3.t = r13 @ cam_1.t + t13
    cam_3.update()


def get_fundamental(cam_1: 'Camera', cam_2: 'Camera'):
    """
    Get the fundamental matrix between two cameras

    In the actually calculation, rotate axes of camera 2, so that external of camera 1 is ``[E|0]``

    (ref: https://sourishghosh.com/2016/fundamental-matrix-from-camera-matrices/)

    Args:
        cam_1 (Camera): Camera instance  whose P, K, R, t, C are all known
        cam_2 (Camera): Camera instance  whose P, K, R, t, C are all known

    Return:
        np.ndarray: the fundamental matrix F, where
            1. x2.T @ F @ x1 = 0
            2. F @ x1 = l2  (epipolar line)
    """
    r12 = cam_2.r @ cam_1.r.T
    t12 = cam_2.r @ (cam_2.c - cam_1.c)
    A = cam_1.k @ r12.T @ t12
    cross = np.array([
        [0, -A[2], A[1]],
        [A[2], 0, -A[0]],
        [-A[1], A[0], 0]
    ])
    F = np.linalg.inv(cam_2.k).T @ r12 @ cam_1.k.T @ cross
    return F


def detect_chessboard(image, corner_number, win_size=5):
    """
    Find the corners on a chessboard

    Args:
        image (np.ndarray): a 2D image.
        corner_number (tuple): the number of points on each side
        win_size (float): for optimisation

    Return:
        np.ndarray: the location of corners in the image. shape (n, 2)
    """
    ret, corners = cv2.findChessboardCorners(
        image, corner_number,
        flags=sum((
            cv2.CALIB_CB_FAST_CHECK,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            ))
    )
    if ret:
        corners_refined = cv2.cornerSubPix(
            image, corners, (win_size, win_size), (-1, -1),
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 0.1
            )
        )
        return np.squeeze(corners_refined)
    else:
        return np.empty((0, 2))


if __name__ == "__main__":
    points = get_points_from_order((8, 6), 'x123')
    points = get_points_from_order((8, 6), '13x2')
    points = get_points_from_order((8, 6), '321x')
    points = get_points_from_order((8, 6), '2x31')
