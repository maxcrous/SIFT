"""
This file contains functions related to assigning a reference
orientation to a keypoint. The central function in this file is
    `assign_reference_orientations`.
"""

from typing import Tuple, List

import numpy as np

import const
from keypoints import Keypoint
from octaves import shift, pixel_dist_in_octave, absolute_sigma


def reference_patch_width(octave_idx: int, sigma: float):
    """ Calculates the width of a neighborhood patch used
        for finding a keypoint's reference orientation. """
    pixel_dist = pixel_dist_in_octave(octave_idx)
    patch_width = (const.reference_patch_width_scalar * sigma) / pixel_dist
    return patch_width


def orientation_to_bin_idx(orientations: np.ndarray) -> np.ndarray:
    """ Converts gradient orientations to histogram bin indices.

    Args:
        orientations: Angles of image gradients in radians with range [0, 2pi].
    Returns:
        orientation histogram bin indices in range [0, const.nr_bins].
    """
    return np.round((const.nr_bins / (2 * np.pi)) * orientations)


# Convolution operations are associative, thus the smoothing filter
# is calculated beforehand and treated as a constant.
smooth_kernel = np.array([1, 1, 1]) / 3
for i in range(const.nr_smooth_iter - 1):
    smooth_kernel = np.convolve(np.array([1, 1, 1]) / 3, smooth_kernel)


def smoothen_histogram(hist: np.array) -> np.array:
    """ Smoothens a histogram with an average filter.
        The filter as defined as multiple convolutions
        with a three-tap box filter [1, 1, 1] / 3.
        See AOS section 4.1.B.

    Args:
        hist: A histogram containing gradient orientation counts.
    Returns:
        hist_smoothed: The histogram after average smoothing.
    """
    pad_amount = round(len(smooth_kernel) / 2)
    hist_pad = np.pad(hist, pad_width=pad_amount, mode='wrap')
    hist_smoothed = np.convolve(hist_pad, smooth_kernel, mode='valid')
    return hist_smoothed


def gradients(octave: np.array) -> Tuple[np.array, np.array]:
    """ Finds the magnitude and orientation of image gradients.

    Args:
         octave: An octave of Gaussian convolved images.
    Returns:
        magnitude: The magnitudes of gradients.
        orientation: The orientation of gradients. Expressed in
            the range [0, 2pi]

    """
    o = octave
    dy = (shift(o, [0, 1, 0]) - shift(o, [0, -1, 0])) / 2
    dx = (shift(o, [0, 0, 1]) - shift(o, [0, 0, -1])) / 2

    # Modulo is to shift range from [-pi, pi] to [0, 2pi]
    magnitudes = np.sqrt(dy ** 2 + dx ** 2)
    orientations = np.arctan2(dy, dx) % (2 * np.pi)

    return magnitudes, orientations


def patch_in_frame(coord: np.array,
                   half_width: float,
                   shape: tuple) -> bool:
    """ Checks whether a square patch falls within the borders of a tensor.

    Args:
        coord: Center coordinate of the patch.
        half_width: Half of the square patch's width.
        shape: Shape of the tensor that contains the patch.
    Returns:
        valid: True if patch is in frame, False if it is not.
    """
    s, y, x = coord.round()
    s_lim, y_lim, x_lim = shape

    valid = (y - half_width > 0
             and y + half_width < y_lim
             and x - half_width > 0
             and x + half_width < x_lim
             and 0 <= s < s_lim)

    return valid


def weighting_matrix(center_offset: np.array,
                     patch_shape: tuple,
                     octave_idx: int,
                     sigma: float,
                     locality: float) -> np.array:
    """ Calculates a Gaussian weighting matrix.
        This matrix determines the weight that gradients
        in a keypoint's neighborhood have when contributing
        to the keypoint's orientation histogram. See AOS section 4,
        Lowe section 5.

    Args:
        center_offset: The keypoint's offset from the patch's center.
        patch_shape: The shape of the patch. The generated weighting
            matrix will need to have the same shape to allow weighting
            by multiplication.
        octave_idx: The index of the octave.
        sigma: The scale of the Difference of Gaussian layer where
               the keypoint was found.
        locality: The locality of the weighting. A higher locality
            is associated with a larger neighborhood of gradients.
            See lambda parameters in AOS section 6 table 4.
    """
    pixel_dist = pixel_dist_in_octave(octave_idx)
    y_len, x_len = patch_shape
    center = np.array(patch_shape) / 2 + center_offset
    y_idxs = np.arange(y_len)
    x_idxs = np.arange(x_len)
    xs, ys = np.meshgrid(y_idxs, x_idxs)
    rel_dists = np.sqrt((xs - center[1]) ** 2 + (ys - center[0]) ** 2)
    abs_dists = rel_dists * pixel_dist
    denom = 2 * ((locality * sigma) ** 2)
    weights = np.exp(-((abs_dists ** 2) / denom))
    return weights


def find_histogram_peaks(hist: np.array) -> List[float]:
    """ Finds peaks in the gradient orientations histogram,
        and returns the corresponding orientations in radians.
        Peaks are the maximum bin and bins that lie within 0.80
        of the mass of the maximum bin. See AOS section 4.1 and
        Lowe section 5. When the modulo operator is used in this
        function, it is to  account for the fact that the first
        and last bin are neighbors, namely, the rotations by 0
        and 2pi radians.

    Args:
        hist: Histogram where each bin represents an orientation, in other
            words, an angle of a gradient. The mass of the bin is determined
            by the number of gradients in the keypoint's local neighborhood
            that have that orientation.
    Returns:
        orientations: The orientations of the peaks in radians. In other words,
            the dominant orientations of gradients in the local neighborhood of
            the keypoint.
    """
    orientations = list()
    global_max = None
    hist_masked = hist.copy()

    for i in range(const.max_orientations_per_keypoint):
        max_idx = np.argmax(hist_masked)
        max_ = hist[max_idx]

        if global_max is None:
            global_max = max_

        if i == 0 or max_ > (0.8 * global_max):
            left = hist[(max_idx - 1) % const.nr_bins]
            right = hist[(max_idx + 1) % const.nr_bins]

            interpol_max_radians = (2 * np.pi * max_idx) / const.nr_bins \
                                    + (np.pi / const.nr_bins) \
                                    * ((left - right) / (left - 2 * max_ + right))

            interpol_max_radians = interpol_max_radians % (2 * np.pi)
            orientations.append(interpol_max_radians)

            # After a peak is found, it and its surrounding bins are masked
            # to enable other peaks to be found with `argmax`.
            for j in range(const.mask_neighbors + 1):
                hist_masked[(max_idx - j) % const.nr_bins] = 0
                hist_masked[(max_idx + j) % const.nr_bins] = 0

    return orientations


def assign_reference_orientations(keypoint_coords: np.array,
                                  gauss_octave: np.array,
                                  octave_idx: int) -> list[Keypoint]:
    """ Assigns dominant local neighborhood gradient orientations to keypoints.
        These dominant orientations are also known as reference orientations.
        A keypoint coordinate may have multiple reference orientations.
        In that case, multiple Keypoint objects are created for that coordinate.
        Reference orientations are used to create rotation invariant descriptors.
        See Lowe section 5, AOS section 4.1.

    Args:
        keypoint_coords: The keypoints' 3D coordinates.
        gauss_octave: An octave of Gaussian convolved images.
        octave_idx: The index of the octave.
    Returns:
        keypoints: A list of keypoints that have been assigned an orientation.
    """
    keypoints = list()
    magnitudes, orientations = gradients(gauss_octave)
    orientation_bins = orientation_to_bin_idx(orientations)
    octave_shape = gauss_octave.shape

    for coord in keypoint_coords:
        s, y, x = coord.round().astype(int)
        sigma = absolute_sigma(octave_idx, s)
        patch_width = reference_patch_width(octave_idx, sigma)
        patch_with_half = round(patch_width / 2)

        if patch_in_frame(coord, patch_with_half, octave_shape):
            orientation_patch = orientation_bins[s,
                                                 y - patch_with_half: y + patch_with_half,
                                                 x - patch_with_half: x + patch_with_half]
            magnitude_patch = magnitudes[s,
                                         y - patch_with_half: y + patch_with_half,
                                         x - patch_with_half: x + patch_with_half]
            patch_shape = magnitude_patch.shape
            center_offset = [coord[1] - y, coord[2] - x]
            weights = weighting_matrix(center_offset, patch_shape, octave_idx, sigma, const.reference_locality)
            contribution = weights * magnitude_patch
            hist, bin_edges = np.histogram(orientation_patch,  bins=const.nr_bins,
                                           range=(0, const.nr_bins), weights=contribution)
            hist = smoothen_histogram(hist)
            dominant_orientations = find_histogram_peaks(hist)

            for orientation in dominant_orientations:
                keypoint = Keypoint(coord=coord, octave_idx=octave_idx, orientation=orientation)
                keypoints.append(keypoint)

    return keypoints
