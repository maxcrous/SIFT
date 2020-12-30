import itertools
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, convolve1d
import functools

import const
# from numba import njit
i = 0 
misbehaved = 0
wellbehaved = 0

def build_gaussian_octaves(img: np.ndarray,
                           nr_octaves: int,
                           dog_intervals_s: int,
                           init_sigma: float) -> List[np.ndarray]:
    """ Builds gaussian octaves, consisting of an image repeatedly
        convolved with a Gaussian kernel.

    Args:
        img: Image used to create the octaves.
        nr_octaves: The total number of octaves to compute.
        dog_intervals_s: The number of layers for which we can find
                         a maxima in a difference of gaussian octave.
        init_sigma: The initial Gaussian sigma used for first octaves first layer.

    Returns:
        octaves: A list of octaves of Gaussian convolved images.
    """
    # Calculate parameters
    imgs_in_octave = dog_intervals_s + 3
    scale_factor_k = 2 ** (1 / dog_intervals_s)

    # Enlarge image for first layer
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    octaves = list()
    next_octave_start = None

    for octave_idx in range(nr_octaves):
        # Start octave with layer from previous octave
        if octave_idx:
            octave = [next_octave_start]
            img = next_octave_start
            next_octave_start = None
        else:
            octave = [img]

        # Repeatedly convolve image with Gaussian
        for layer in range(imgs_in_octave - 1):
            mod_sigma = init_sigma * (scale_factor_k ** layer)
            img = gaussian_filter(img, mod_sigma)

            # When sigma has doubled, save layer for next octave start
            if mod_sigma > init_sigma * 2 and next_octave_start is None:
                next_octave_start = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            octave.append(img)

        octave = np.array(octave)
        octaves.append(octave)

    return octaves


def build_dog_octave(gauss_octave: np.ndarray) -> np.ndarray:
    """ Builds a Difference of Gaussian octave.

    Args:
        gauss_octave: An octave of Gaussian convolved images.

    Returns:
        octave: An octave of Difference of Gaussian images.
    """

    dog_octave = list()

    for layer_idx, layer in enumerate(gauss_octave):
        if layer_idx:
            previous_layer = gauss_octave[layer_idx - 1]
            dog = layer - previous_layer
            dog_octave.append(dog)

    return np.array(dog_octave)


def find_dog_extrema(dog_octave: np.ndarray) -> np.ndarray:
    """ Finds extrema in a Difference of Gaussian octave.

    Args:
        dog_octave: An octave of Difference of Gaussian images.

    Returns:
        extrema_coords: The Difference of Gaussian extrema coordinates.
    """
    shifts = list(itertools.product([-1, 0, 1], repeat=3))
    shifts.remove((0, 0, 0))

    diffs = list()
    for x, y, z in shifts:
        padded = np.pad(dog_octave, 1)
        shifted = padded[1 + x: -1 + x if x != 1 else None,
                         1 + y: -1 + y if y != 1 else None,
                         1 + z: -1 + z if z != 1 else None]

        diff = dog_octave - shifted
        diffs.append(diff)

    diffs = np.array(diffs)
    maxima = np.where((diffs > 0).all(axis=0))
    minima = np.where((diffs < 0).all(axis=0))
    extrema_coords = np.concatenate((maxima, minima), axis=1)

    return extrema_coords


def get_deriv_func():
    """ Gets the appropriate funcion for taking an image derivative. """
    if const.deriv_func == 'simple':
        gradient_filter = np.array([1, 0, -1])
        deriv_func = functools.partial(convolve1d, weights=gradient_filter)
    else:
        deriv_func = sobel

    return deriv_func


def get_derivatives(octave: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculates the first and second order s,x,y derivatives for an octave.

    Args:
        octave: An octave of Difference of Gaussian images.

    Returns:
        derivs: The s, x, and y derivative of the octave.
        second_derivs: The ss, sx, sy, xs, xx, xy, ys, yx, yy derivatives of the octave.
    """
    derivs = list()
    second_derivs = list()

    deriv_func = get_deriv_func()

    for axis in range(3):
        derivs.append(deriv_func(octave, axis=axis) / 8 )

    for deriv in derivs:
        for axis in range(3):
            second_derivs.append(deriv_func(deriv, axis=axis) / 8)

    derivs = np.array(derivs)
    second_derivs = np.array(second_derivs)

    return derivs, second_derivs


# @njit
def get_extremum_offset(deriv: np.ndarray,
                        second_deriv: np.ndarray) -> np.ndarray:
    """ Calculates the offset of the extremum.

    Args:
        deriv: The 1x3 derivatives of the extremum.
        second_deriv: 3x3 Hessian of the extremum.

    Returns:
        offset: The offset from the extremum to the interpolated extremum.
    """
    global misbehaved
    global wellbehaved
    second_deriv = second_deriv.reshape((3, 3))
    second_deriv_inv = np.linalg.inv(second_deriv)
    offset = -np.dot(second_deriv_inv, deriv)
    if abs(offset[1]) > 0.5 or abs(offset[2]) > 0.5 or offset[0] > 0.5:
        misbehaved += 1
    else:
        wellbehaved += 1

    return offset


# @njit
def interpolate_extremum(extremum_coord: tuple,
                         extremum_val: float,
                         deriv: np.ndarray,
                         second_deriv: np.ndarray) -> Tuple[np.ndarray, float]:
    """ Interpolates the coordinate and value of an extremum in a
        Difference of Gaussian octave. Interpolation is done with a
        Taylor expansion to fit a 3D quadratic function to the local sample.
        The new extremum is where this function's derivative equals 0.

    Args:
        extremum_coord: The location of the extremum in a Difference of Gaussian octave.
        extremum_val: The extremum value.
        deriv: The extremum first derivatives.
        second_deriv: The extremum second derivatives.
    """
    extremum_coord = np.array(extremum_coord)
    offset = get_extremum_offset(deriv, second_deriv)
    val_change = (1 / 2) * np.dot(deriv, offset)
    interpol_extremum_coord = extremum_coord + offset
    interpol_extremum_val = extremum_val + val_change
    return interpol_extremum_coord, interpol_extremum_val


# @njit
def pass_edge_test(second_deriv: np.ndarray) -> bool:
    """ Eliminates keypoints along edges for stability.
        Returns true when keypoint passes edge test, false when it fails.

    Args:
        second_deriv: The hessian of the keypoint.

    Returns:
        passes: whether the keypoint passes the edge test.
    """
    e = const.eps
    r = const.r
    xy_hessian = second_deriv.reshape((3, 3))[1:, 1:].copy()
    trace = np.trace(xy_hessian)
    det = np.linalg.det(xy_hessian) + e
    threshold = ((r + 1) ** 2) / r
    return (trace ** 2) / (det + e) < threshold


def find_keypoints(extrema: np.ndarray, dog_octave: np.ndarray) -> np.ndarray:
    """ Finds keypoints among all candidate extrema.

    Args:
        extrema: The Difference of Gaussian extrema coordinates.
        dog_octave: Difference of Gaussian images.

    Returns:
        keypoint_coords: Interpolated coordinates of keypoints
    """
    keypoint_coords = list()
    derivs, second_derivs = get_derivatives(dog_octave)

    for s, x, y in extrema.T:
        extremum_coord = tuple([s, x, y])
        extremum = dog_octave[extremum_coord]

        if abs(extremum) > const.magnitude_threshold:
            deriv = derivs[:, s, x, y].copy()
            second_deriv = second_derivs[:, s, x, y].copy()
            extremum_coord, extremum = interpolate_extremum(extremum_coord, extremum, deriv, second_deriv)

            if pass_edge_test(second_deriv):
                keypoint_coords.append(extremum_coord)

    return np.array(keypoint_coords)


def assign_orientation(keypoint_coords, gauss_octave):
    return None


def main():
    nr_octaves = const.nr_octaves
    init_sigma = const.init_sigma
    dog_intervals_s = const.dog_intervals_s

    img = cv2.imread('house.jpg', flags=cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    gauss_octaves = build_gaussian_octaves(img, nr_octaves, dog_intervals_s, init_sigma)

    for gauss_octave in gauss_octaves:
        dog_octave = build_dog_octave(gauss_octave)
        extremas = find_dog_extrema(dog_octave)
        keypoint_coords = find_keypoints(extremas, dog_octave)


class Keypoint:
    def __init__(self, coord, octave_idx):
        self.coord = coord
        self.octave = octave_idx
        self.orientation = np.histogram


main()
print(misbehaved)
print(wellbehaved)
