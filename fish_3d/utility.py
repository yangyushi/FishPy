import cv2
import numpy as np
from tqdm import tqdm
from numba import njit
from typing import List
from joblib import Parallel, delayed
from scipy import ndimage
import matplotlib.pyplot as plt
from . import ray_trace

dpi = 150


def see_corners(image_file, corner_number=(23, 15)):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
            gray, corner_number,
            flags=sum((
                cv2.CALIB_CB_FAST_CHECK,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                )))
    corners = np.squeeze(corners)
    plt.imshow(np.array(gray), cmap='gray')
    plt.plot(*corners.T[:2], color='tomato')
    plt.xlim(corners[:, 0].min() - 100, corners[:, 0].max() + 100)
    plt.ylim(corners[:, 1].min() - 100, corners[:, 1].max() + 100)
    plt.scatter(*corners[0].T, color='tomato')
    plt.axis('off')
    plt.show()


def plot_reproject(
        image, features, pos_3d, camera,
        filename=None, water_level=0, normal=(0, 0, 1)
        ):
    fig = plt.figure(figsize=(image.shape[1]/dpi, image.shape[0]/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    for point in pos_3d:
        xy = ray_trace.reproject_refractive(point, camera)
        ax.scatter(*xy, color='tomato', marker='+', lw=1, s=128)

    ax.scatter(
            features[0], features[1],
            color='w', facecolor='none', alpha=0.5
            )

    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.gcf().set_frameon(False)
    plt.gcf().axes[0].get_xaxis().set_visible(False)
    plt.gcf().axes[0].get_yaxis().set_visible(False)
    plt.axis('off')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_reproject_with_roi(
        image, roi, features, pos_3d, camera,
        filename=None, water_level=0, normal=(0, 0, 1)
        ):
    fig = plt.figure(figsize=(image.shape[1]/dpi, image.shape[0]/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    for point in pos_3d:
        xy = ray_trace.reproject_refractive(point, camera)
        ax.scatter(*xy, color='tomato', marker='+', lw=1, s=128)

    ax.scatter(
            features[0] + roi[1].start, features[1] + roi[0].start,
            color='w', facecolor='none', alpha=0.5
            )

    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.gcf().set_frameon(False)
    plt.gcf().axes[0].get_xaxis().set_visible(False)
    plt.gcf().axes[0].get_yaxis().set_visible(False)
    plt.axis('off')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_clusters(image, threshold, min_size, roi):
    """
    apply threshold to image and label disconnected part
    small labels (size < min_size) were erased
    the coordinates of labels were returned, shape: (label_number, 3)
    """
    binary = image > (image[roi].max() * threshold)
    mask = np.zeros(image.shape, dtype=int)
    mask[roi] = 1
    labels, _ = ndimage.label(binary)
    labels *= mask
    counted = np.bincount(labels.ravel())
    noise_indice = np.where(counted < min_size)
    mask = np.in1d(labels, noise_indice).reshape(image.shape)
    labels[mask] = 0
    labels, _ = ndimage.label(labels)
    clusters = []
    for value in np.unique(labels.ravel()):
        if value > 0:
            cluster = np.array(np.where(labels == value)).T
            clusters.append(cluster)
    return clusters


def plot_epl(line: List[float], image: np.ndarray):
    """
    Get the scatters of an epipolar line *inside* an image
    The images is considered as being stored in (row [y], column [x])
    """
    u = np.linspace(0, image.shape[1], 1000)
    v = -(line[2] + line[0] * u) / line[1]
    mask = np.ones(len(u), dtype=bool)
    mask[v < 0] = False
    mask[v >= image.shape[0]] = False
    uv = np.vstack([u, v]).T[mask]
    uv_dst = uv
    mask = np.ones(len(uv_dst), dtype=bool)
    mask[uv_dst[:, 0] < 0] = False
    mask[uv_dst[:, 0] >= image.shape[1]] = False
    mask[uv_dst[:, 1] < 0] = False
    mask[uv_dst[:, 1] >= image.shape[0]] = False
    epl = uv_dst[mask]
    return epl.T


def get_ABCD(corners: np.array, width: int, excess_rows: int) -> dict:
    """
    :param corners: the coordinates of chessboard corners
                    found by cv2.findChessboardCorners, shape (n, 2)
    :param width: the number of corners in a row
    :param excess_rows: if the chessboard is make of (m, n) corners (m > n)
                        then n is *width*, excess_rows = m - n
    what is ABCD?
    C +-------+ D
      |       |
      |       |
      |       |
    A +-------+ B
    (AB // CD, AC // BD)
    """
    points_Ah, points_Bh, points_Ch, points_Dh = [], [], [], []
    for er in range(excess_rows + 1):
        start_AB = er * width
        start_CD = start_AB + width * (width - 1)
        offset = width - 1
        A = corners[start_AB].tolist()
        B = corners[start_AB + offset].tolist()
        C = corners[start_CD].tolist()
        D = corners[start_CD + offset].tolist()
        # collect the points
        Ah = A + [1]
        Bh = B + [1]
        Ch = C + [1]
        Dh = D + [1]
        points_Ah.append(Ah)  # (n, 3)
        points_Bh.append(Bh)  # (n, 3)
        points_Ch.append(Ch)  # (n, 3)
        points_Dh.append(Dh)  # (n, 3)
    ABCD = {'A': np.array(points_Ah),
            'B': np.array(points_Bh),
            'C': np.array(points_Ch),
            'D': np.array(points_Dh)}
    return ABCD


def get_affinity(abcd: dict) -> np.array:
    """
    get the affinity matrix from a set of corners measured from
    a chessboard image
    :param abcd: a dict storing the measured coordinates of chessboard corners
    what is ABCD?
    C +-------+ D
      |       |
      |       |
      |       |
    A +-------+ B
    """
    lAB = np.cross(abcd['A'], abcd['B'])  # (n, 3)
    lCD = np.cross(abcd['C'], abcd['D'])  # (n, 3)
    lAC = np.cross(abcd['A'], abcd['C'])  # (n, 3)
    lBD = np.cross(abcd['B'], abcd['D'])  # (n, 3)
    points_inf_1 = np.cross(lAB, lCD)
    points_inf_1 = (points_inf_1/points_inf_1[:, -1][:, None])[:, :2]  # sorry
    points_inf_2 = np.cross(lAC, lBD)
    points_inf_2 = (points_inf_2/points_inf_2[:, -1][:, None])[:, :2]  # (n, 2)

    points_inf = np.vstack(
            (points_inf_1, points_inf_2)  # [(n, 2), (n, 2)] -> (2n, 2)
            )
    a, b = np.polyfit(*points_inf.T, deg=1)
    H_aff = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [a/b, -1/b, 1]
    ])
    return H_aff


def get_similarity(abcd: dict, H_aff: np.array) -> np.array:
    """
    get the similarity matrix from a set of corners measured from
    a chessboard image
    :param abcd: a dict storing the measured coordinates of chessboard corners
    :param H_aff: affinity that makes coordinates affinely recitified
    what is ABCD?
    C +-------+ D
      |       |
      |       |
      |       |
    A +-------+ B
    """
    abcd_aff = {}
    for letter in abcd:
        aff_rec = H_aff @ abcd[letter].T  # (3, n)
        aff_rec = aff_rec / aff_rec[-1]
        abcd_aff.update({letter: aff_rec.T})
    lAB = np.cross(abcd_aff['A'], abcd_aff['B'])  # (n, 3)
    lAC = np.cross(abcd_aff['A'], abcd_aff['C'])  # (n, 3)
    lAD = np.cross(abcd_aff['A'], abcd_aff['D'])  # (n, 3)
    lBC = np.cross(abcd_aff['B'], abcd_aff['C'])  # (n, 3)
    lAB = (lAB / lAB[:, -1][:, None])[:, :2]
    lAC = (lAC / lAC[:, -1][:, None])[:, :2]
    lAD = (lAD / lAD[:, -1][:, None])[:, :2]
    lBC = (lBC / lBC[:, -1][:, None])[:, :2]
    # solve circular point constrain from 2 pp lines
    s11_ensemble, s12_ensemble = [], []
    M = np.empty((2, 3))
    for i in range(len(abcd['A'])):  # for different "point A"
        M[0, 0] = lAB[i, 0] * lAC[i, 0]
        M[0, 1] = lAB[i, 0] * lAC[i, 1] + lAB[i, 1] * lAC[i, 0]
        M[0, 2] = lAB[i, 1] * lAC[i, 1]
        M[1, 0] = lAD[i, 0] * lBC[i, 0]
        M[1, 1] = lAD[i, 0] * lBC[i, 1] + lAD[i, 1] * lBC[i, 0]
        M[1, 2] = lAD[i, 1] * lBC[i, 1]
        s11, s12 = np.linalg.solve(M[:, :2], -M[:, -1])
        s11_ensemble.append(s11)
        s12_ensemble.append(s12)
    S = np.array([
            [np.mean(s11_ensemble), np.mean(s12_ensemble)],
            [np.mean(s12_ensemble), 1],
        ])
    S = S / max(s11, 1)  # not scaling up the coordinates
    K = np.linalg.cholesky(S)
    Hs_inv = np.array([  # from similar to affine
        [K[0, 0], K[0, 1], 0],
        [K[1, 0], K[1, 1], 0],
        [0, 0, 1]
        ])
    return np.linalg.inv(Hs_inv)


def get_corners(image: np.array, rows: int, cols: int, camera_model=None):
    """
    use findChessboardCorners in opencv to get coordinates of corners
    """
    ret, corners = cv2.findChessboardCorners(
        image, (rows, cols),
        flags=sum((
            cv2.CALIB_CB_FAST_CHECK,
            cv2.CALIB_CB_ADAPTIVE_THRESH
        ))
    )
    corners = np.squeeze(corners)  # shape (n, 1, 2) -> (n, 2)

    # undistorting points does make l_inf fit better
    if not isinstance(camera_model, type(None)):
        corners = camera_model.undistort_points(corners, want_uv=True)

    return corners


def get_homography_image(image: np.array, rows: int, cols: int, camera_model=None):
    """
    get the homography transformation
    from an image with a chess-board
    image: 2d numpy array
    rows: the number of *internal corners* inside each row
    cols: the number of *internal corners* inside each column
    camera_model: (optional) a Camera instance that stores the
                  distortion coefficients of the lens
    """
    length = max(rows, cols)  # length > width
    width = min(rows, cols)

    corners = get_corners(image, width, length, camera_model)
    excess_rows = length - width

    abcd = get_ABCD(corners, width, excess_rows)

    H_aff = get_affinity(abcd)
    H_sim = get_similarity(abcd, H_aff)

    return H_sim @ H_aff


def get_homography(camera: 'Camera', angle_num=10):
    """
    get the homography that simiarly recover the 2d image perpendicular to z-axis
    :param camera: a Camera instance of current camera
    :param angle_num: a virtual chessboard is rotated angle_num times for calculation
    """
    angles = np.linspace(0, np.pi/2, angle_num)  # rotation_angle

    abcd = {
        'A': np.empty((angle_num, 3)),
        'B': np.empty((angle_num, 3)),
        'C': np.empty((angle_num, 3)),
        'D': np.empty((angle_num, 3)),
    }

    for i, t in enumerate(angles):
        R = np.array((  # rotation matrix pp z-axis
            (np.cos(t), -np.sin(t), 0),
            (np.sin(t), np.cos(t), 0),
            (0, 0, 1)
        ), dtype=np.float32)

        abcd_3d = np.array((  # shape (3 dim, 4 corners)
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
        ), dtype=np.float32).T

        abcd_3d[:2, :] -= abcd_3d[:2, :].mean(axis=1)[:, np.newaxis]

        abcd_3d_rot = R @ abcd_3d  # shape (3 dim, 4 corners)

        abcd_3d_rot_h = np.vstack((abcd_3d_rot, np.ones((1, 4))))  # shape (4 dim, 4 corners)

        abcd_2dh = camera.p @ abcd_3d_rot_h
        abcd_2dh = (abcd_2dh / abcd_2dh[-1, :]).T  # shape (4 corners, 3 dim)

        abcd['A'][i] = abcd_2dh[0]
        abcd['B'][i] = abcd_2dh[1]
        abcd['C'][i] = abcd_2dh[2]
        abcd['D'][i] = abcd_2dh[3]

    H_aff = get_affinity(abcd)
    H_sim = get_similarity(abcd, H_aff) @ H_aff

    return H_sim


def update_orientation(
        orientations: np.array, locations: np.array, H: np.array, length=10
        ):
    """
    Calculate the orientatin after applying homography H
    This is function is used to get a 'recitified' orientation
    :param orientation: angles of the fish, sin(angle) --> x very sadly
    :param locatons: xy positons of fish in the image, not row-col
    :param H: the homography matrix
    :param length: length of the orientation bar
    """
    orient_rec = []
    for o, xy in zip(orientations, locations):
        p1h = np.array((
            xy[0] - length * np.sin(o),
            xy[1] - length * np.cos(o), 1
        ))
        p2h = np.array((
            xy[0] + length * np.sin(o),
            xy[1] + length * np.cos(o), 1
        ))
        p1h = H @ p1h
        p2h = H @ p2h
        p1 = (p1h / p1h[-1])[:2]
        p2 = (p2h / p2h[-1])[:2]
        o_rec = np.arctan2(*(p2 - p1))
        orient_rec.append(o_rec)
    orient_rec = np.array(orient_rec)
    orient_rec[orient_rec < 0] += np.pi
    orient_rec[orient_rec > np.pi] -= np.pi
    return orient_rec


def get_orient_line(locations, orientations, length=10):
    """
    get the line for plot the orientations
    :param locations: shape (n, 2)
    :param orientations: shape (n, )
    """
    unit_vector = np.array((np.sin(orientations), np.cos(orientations)))
    oline_1 = locations - length * unit_vector
    oline_2 = locations + length * unit_vector

    olines_x = []
    olines_y = []

    for i in range(oline_1.shape[1]):
        olines_x += [oline_1[0, i], oline_2[0, i], np.nan]
        olines_y += [oline_1[1, i], oline_2[1, i], np.nan]
    return np.array((olines_x, olines_y))


@njit(fastmath=True)
def polar_chop(image, H_sim, centre, radius, n_angle, n_radius, dist_coef, k):
    """
    Chop an image in the polar coordinates
    return the chopped result as a labelled image
    :param image: 2d image as a numpy array
    :param H_sim: a homography (3 x 3 matrix) to similarly rectify the image
    :param centre: origin of the polar coordinate system
    :param radius: maximum radius in the polar coordinate system
    :param n_angle: number of bins in terms of angle
    :param n_radius: number of bins in terms of radius
    :param dist_coef: distortion coefficients of the camera, shape (5, )
                        k1, k2, p1, p2, k3 (from opencv by default)
    :param k: camera calibration matrix (bible, P155)
    """
    # setting up bin edges
    be_angle = np.linspace(0, 2 * np.pi, n_angle+1)  # bin_edge
    r0 = np.sqrt(radius**2 / (n_angle * (n_radius-1) + 1))
    be_radius = np.empty(n_radius+1)
    be_radius[0] = 0
    be_radius[1] = r0
    for i in range(2, n_radius+1):
        be_radius[i] = np.sqrt(((i-1) * n_angle + 1))* r0
    be_r2 = be_radius ** 2

    # getting camera parameters
    k1, k2, p1, p2, k3 = dist_coef
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

    result = np.empty(image.shape, dtype=np.uint64)
    for x in range(image.shape[1]):  # x -> col
        for y in range(image.shape[0]):  # y -> row!
            # undistort x, y
            x0 = (x - cx) / fx
            y0 = (y - cy) / fy
            x_ud, y_ud = x0, y0  # ud -> undistorted
            for _ in range(5):  # iter 5 times
                r2 = x_ud**2 + y_ud ** 2
                k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
                delta_x = 2 * p1 * x_ud*y_ud + p2 * (r2 + 2 * x_ud**2)
                delta_y = p1 * (r2 + 2 * y_ud**2) + 2 * p2 * x_ud*y_ud
                x_ud = (x0 - delta_x) * k_inv
                y_ud = (y0 - delta_y) * k_inv
            x_ud = x_ud * fx + cx
            y_ud = y_ud * fy + cy

            # similar rectification
            xyh = np.array([x_ud, y_ud, 1], dtype=np.float64)
            xyh_sim = H_sim @ xyh
            xy_sim = (xyh_sim / xyh_sim[-1])[:2]  # similar trasnformed
            x_sim, y_sim = xy_sim - centre

            # work in polar coordinates
            t = np.arctan2(y_sim, x_sim) + np.pi  # theta
            r2 = x_sim**2 + y_sim**2

            # label the image
            if r2 <= be_r2[1]:
                result[y, x] = 1
            elif r2 <= be_r2[-1]:
                idx_angle, idx_radius = 0, 0
                for i, a in enumerate(be_angle[1:]):
                    if (t < a) and (t >= be_angle[i]):
                        idx_angle = i
                for j, r2e in enumerate(be_r2[1:]):
                    if (r2 < r2e) and (r2 >= be_r2[j]):
                        idx_radius = j
                idx_group = idx_angle + (idx_radius - 1) * n_angle + 2
                result[y, x] = idx_group
            else:
                result[y, x] = 0
    return result


def get_indices(labels):
    indices = []
    for val in set(np.ravel(labels)):
        if val > 0:
            indices.append(
                np.array(np.where(labels.ravel() == val)[0])
            )
    return indices


def box_count_polar_image(image, indices):
    """
    :param image: the image taken by the camera without undistortion
    :param labels: labelled image specifying different box regions
    """
    invert = image.max() - image.ravel()
    intensities = np.empty(len(indices))
    for i, idx in enumerate(indices):
        pixels = invert[idx]
        intensities[i] = np.mean(pixels)
    return np.std(intensities), np.min(intensities), np.mean(intensities)


def box_count_polar_video(video, labels, cores=2, report=True):
    if report:
        to_iter = tqdm(video)
    else:
        to_iter = video
    indices = get_indices(labels)
    results = Parallel(n_jobs=cores, require='sharedmem')(
        delayed(
            lambda x: box_count_polar_image(x, indices)
        )(frame) for frame in to_iter
    )
    return np.array(results).T  # (3, n)

