import cv2
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from typing import List
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


def get_homography(image: np.array, rows: int, cols: int, camera_model=None):
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
