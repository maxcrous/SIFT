"""
This file contains functions related to generating and manipulating
the Gaussian octaves and Difference of Gaussian octaves of an image.
The central functions in this file are
    `build_gaussian_octaves`
    and
    `build_dog_octave`
"""

import itertools
import math
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

import const


def progression_sigma(layer_idx: int) -> float:
    """ Calculates the Gaussian blur's standard deviation required
        to move from (layer_idx - 1) -> layer_idx in scale-space.
        See AOS section 2.2. This ensures each octave covers a
        doubling of the blurring or sigma, and that the generated
        layer's sigmas obey the following formula:

        layer_sigma = (octave_idx / min_pixel_dist) * min_sigma * 2 ** (layer_idx / scales_per_octave)

    Args:
        layer_idx: The index of a layer.
    Returns:
        sigma: The Gaussian blur filter's standard deviation required to
            to move from (layer_idx - 1) -> layer_idx.
    """
    sigma = (const.min_sigma / const.min_pixel_dist) \
            * math.sqrt(2 ** (2 * layer_idx / const.scales_per_octave)
                        - 2 ** (2 * (layer_idx - 1) / const.scales_per_octave))
    return sigma


def absolute_sigma(octave_idx: int,
                   layer_idx: int) -> float:
    """ Calculates the Gaussian blur's standard deviation
        associated with the blurring of a layer. See AOS section
        2.2. While progression_sigma provides the relative sigma
        required to move from one layer to the next, this function
        provides the layer's absolute sigma. In other words, the
        level of blurring required to move from the original image
        to this layer in scale-space.

    Args:
        octave_idx: The index of an octave. Here the first octave
            with index 0 is the octave of the 2x upsampled original
            image with pixel distance 0.5.
        layer_idx: The index of a layer.
    Returns:
        sigma: The Gaussian blur filter's standard deviation required to
            to move from (layer_idx - 1) -> layer_idx.
    """
    pixel_dist = pixel_dist_in_octave(octave_idx)
    sigma = (pixel_dist / const.min_pixel_dist) * const.min_sigma * 2 ** (layer_idx / const.scales_per_octave)
    return sigma


def build_gaussian_octaves(img: np.ndarray) -> List[np.ndarray]:
    """ Builds Gaussian octaves, consisting of an image repeatedly
        convolved with a Gaussian kernel. See AOS section 2.2.

    Args:
        img: Image used to create the octaves.
    Returns:
        octaves: A list of octaves of Gaussian convolved images.
            Here, each octave is a [s, y, x] 3D tensor.
    """
    layers_per_octave = const.scales_per_octave + const.auxiliary_scales
    octaves = list()
    previous_octave = None

    for octave_idx in range(const.nr_octaves):

        # Start the first octave with the 2x upsampled input image.
        # All other octaves start with the 2x downsampled previous
        # octave's second from last layer.
        if octave_idx == 0:
            img = cv2.resize(img, None, fx=const.first_upscale, fy=const.first_upscale, interpolation=cv2.INTER_LINEAR)
            img = gaussian_filter(img, const.init_sigma)
            octave = [img]
        else:
            img = cv2.resize(previous_octave[-2], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            octave = [img]

        # Convolve layers with gaussians to generate successive layers.
        # The previous octave's[-2] upsampled image is considered layer
        # index 0, so indexing starts at 1
        for layer_idx in range(1, layers_per_octave):
            sigma = progression_sigma(layer_idx)
            img = gaussian_filter(img, sigma)
            octave.append(img)

        previous_octave = octave
        octave = np.array(octave)
        octaves.append(octave)

    return octaves


def build_dog_octave(gauss_octave: np.ndarray) -> np.ndarray:
    """ Builds a Difference of Gaussian octave.

    Args:
        gauss_octave: An octave of Gaussian convolved images.
    Returns:
        dog_octave: An octave of Difference of Gaussian images.
    """
    dog_octave = list()

    for layer_idx, layer in enumerate(gauss_octave):
        if layer_idx:
            previous_layer = gauss_octave[layer_idx - 1]
            dog = layer - previous_layer
            dog_octave.append(dog)

    return np.array(dog_octave)


def shift(array: np.ndarray,
          shift_spec: list or tuple) -> np.ndarray:
    """ Takes a 3D tensor and shifts it in a specified direction.

    Args:
        array: The 3D array that is to be shifted.
        shift_spec: The shift specification for each of
            the 3 axes. E.g., [1, 0, 0] will make the
            element (x,x,x) equal element (x+1, x+1, x+1) in
            the original image, effectively shifting the
            image "to the left", along the first axis.
    Returns:
        shifted: The shifted array.
    """
    padded = np.pad(array, 1, mode='edge')
    s, y, x = shift_spec
    shifted = padded[1 + s: -1 + s if s != 1 else None,
                     1 + y: -1 + y if y != 1 else None,
                     1 + x: -1 + x if x != 1 else None]
    return shifted


def find_dog_extrema(dog_octave: np.ndarray) -> np.ndarray:
    """ Finds extrema in a Difference of Gaussian octave.
        This is achieved by subtracting a cell by all it's
        direct (including diagonal) neighbors, and confirming
        all differences have the same sign.

    Args:
        dog_octave: An octave of Difference of Gaussian images.
    Returns:
        extrema_coords: The Difference of Gaussian extrema coordinates.
    """
    shifts = list(itertools.product([-1, 0, 1], repeat=3))
    shifts.remove((0, 0, 0))

    diffs = list()
    for shift_spec in shifts:
        shifted = shift(dog_octave, shift_spec)
        diff = dog_octave - shifted
        diffs.append(diff)

    diffs = np.array(diffs)
    maxima = np.where((diffs > 0).all(axis=0))
    minima = np.where((diffs < 0).all(axis=0))
    extrema_coords = np.concatenate((maxima, minima), axis=1)

    return extrema_coords


def derivatives(dog_octave: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculates the first and second order s, y, x derivatives for an octave.

    Args:
        dog_octave: An octave of Difference of Gaussian images.
    Returns:
        derivs: The s, y, and x derivatives of the octave.
        second_derivs: The ss, yy, xx, sy, sx, yx derivatives of the octave.
            Provided in the format of a flattened Hessian for convenient
            indexing and reshaping.
    """
    o = dog_octave

    ds = (shift(o, [1, 0, 0]) - shift(o, [-1, 0, 0])) / 2
    dy = (shift(o, [0, 1, 0]) - shift(o, [0, -1, 0])) / 2
    dx = (shift(o, [0, 0, 1]) - shift(o, [0, 0, -1])) / 2

    dss = (shift(o, [1, 0, 0]) + shift(o, [-1, 0, 0]) - 2 * o)
    dyy = (shift(o, [0, 1, 0]) + shift(o, [0, -1, 0]) - 2 * o)
    dxx = (shift(o, [0, 0, 1]) + shift(o, [0, 0, -1]) - 2 * o)

    dsy = (shift(o, [1, 1, 0]) - shift(o, [1, -1, 0]) - shift(o, [-1, 1, 0]) + shift(o, [-1, -1, 0])) / 4
    dsx = (shift(o, [1, 0, 1]) - shift(o, [1, 0, -1]) - shift(o, [-1, 0, 1]) + shift(o, [-1, 0, -1])) / 4
    dyx = (shift(o, [0, 1, 1]) - shift(o, [0, 1, -1]) - shift(o, [0, -1, 1]) + shift(o, [0, -1, -1])) / 4

    derivs = np.array([ds, dy, dx])
    second_derivs = np.array([dss, dsy, dsx,
                              dsy, dyy, dyx,
                              dsx, dyx, dxx])
    return derivs, second_derivs


def pixel_dist_in_octave(octave_idx: int) -> float:
    """ Calculates the distance between adjacent pixels in an octave.
        As each octave starts with a 2x downsampled image, each successive
        octave doubles the pixel distance.

    Args:
        octave_idx: The index of the octave.
    Returns:
        The distance between adjacent pixels in an octave.
    """
    return const.min_pixel_dist * (2 ** octave_idx)
