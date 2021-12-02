"""
This file contains functions related to assigning a descriptor
to a keypoint. The central function in this file is
    `assign_descriptor`.
"""

import numpy as np

import const
from keypoints import Keypoint
from octaves import pixel_dist_in_octave
from reference_orientation import gradients, patch_in_frame, weighting_matrix


def hist_centers() -> np.ndarray:
    """ Calculates relative coordinates of histogram centers within a descriptor patch.

    Returns:
        centers: Relative coordinates of histogram centers, in format (16, 2)
    """
    xs = list()
    ys = list()

    bin_width = (2 * const.descriptor_locality) / const.nr_descriptor_histograms
    hist_center_offset = bin_width / 2
    start_coord = -const.descriptor_locality + hist_center_offset

    for row_idx in range(const.nr_descriptor_histograms):
        for col_idx in range(const.nr_descriptor_histograms):
            y = start_coord + bin_width * row_idx
            x = start_coord + bin_width * col_idx
            ys.append(y)
            xs.append(x)

    centers = np.array([xs, ys]).T
    return centers


# All patches have the same relative histogram centers,
# so calculate them beforehand and treat as constants.
histogram_centers = hist_centers()


def relative_patch_coordinates(center_offset: list,
                               patch_shape: tuple,
                               pixel_dist: float,
                               sigma: float,
                               keypoint_orientation: float) -> np.ndarray:
    """ Calculates the coordinates of pixels in a descriptor patch,
        relative to the keypoint. Keypoints have an orientation and
        therefore introduce an oriented x and y axis. This is why
        the relative coordinates are the result of a rotation.
        See Lowe section 5 and AOS section 4.2.

    Args:
        center_offset: The keypoint's offset from the patch's center.
        patch_shape: The shape of a descriptor patch including padding.
        pixel_dist: The distance between adjacent pixels.
        sigma: The scale of layer where the keypoint was found.
        keypoint_orientation: The orientation of the keypoint in radians.
    Returns:
        rel_coords: The y & x coordinates of pixels in a descriptor patch
            relative to the keypoint's location and orientation.
    """
    y_len, x_len = patch_shape
    center = np.array(patch_shape) / 2 + center_offset
    y_idxs = np.arange(y_len)
    x_idxs = np.arange(x_len)
    xs, ys = np.meshgrid(y_idxs, x_idxs)

    # Coordinates are rotated to align with the keypoint's orientation.
    rel_xs = ((xs - center[1]) * np.cos(keypoint_orientation)
              + (ys - center[0]) * np.sin(keypoint_orientation)) / (sigma / pixel_dist)

    rel_ys = (-(xs - center[1]) * np.sin(keypoint_orientation)
              + (ys - center[0]) * np.cos(keypoint_orientation)) / (sigma / pixel_dist)

    rel_coords = np.array([rel_xs, rel_ys])
    return rel_coords


def mask_outliers(magnitude_patch: np.ndarray,
                  rel_patch_coords: np.ndarray,
                  threshold: float,
                  axis: int = 0) -> np.ndarray:
    """ Masks outliers in a patch. Here, an outlier has a distance
        from the patch's center keypoint along the y or x axis that
        is larger than the threshold.

    Args:
        magnitude_patch: The gradient magnitudes in the patch.
        rel_patch_coords: The y & x coordinates of pixels in a descriptor patch
            relative to the keypoint's location and potentially orientation.
        threshold: Distance in y and x after which a point is masked to 0.
        axis: The axis along which the max between y & x is found.
    Returns:
        magnitude_patch: The  gradient magnitudes in the patch after masking.
    """
    mask = np.max(np.abs(rel_patch_coords), axis=axis) <= threshold
    magnitude_patch = magnitude_patch * mask
    return magnitude_patch


def interpolate_2d_grid_contribution(magnitude_path: np.ndarray,
                                     coords_rel_to_hist: np.ndarray):
    """ Interpolates gradient contributions to surrounding histograms.
        In other words: Calculates to what extent gradients in a descriptor
        patch contribute to a histogram, based on the gradient's pixel's
        y & x distance to that histogram's location. See AOS section 4.2
        and figure 10 and Lowe section 6. This function performs the
        interpolation for all histograms at once via broadcasting.

    Args:
        magnitude_path: The gradient magnitudes in a descriptor patch, used to
            weigh gradient contributions. For the standard configuration,
            this array is of shape (2, 32, 32) with semantics (y_or_x, patch_row, patch_col).
        coords_rel_to_hist: The coordinates of pixels in a descriptor patch,
            relative to a histograms location. For the standard configuration,
            this array is of shape (2, 16, 32, 32) after axes swap, with
            semantics (y_or_x, hist_idx, patch_row, patch_col).
    Returns:
        magnitude_path: The gradient magnitudes in a descriptor patch after
            interpolating their contributions for each histogram.
            For the standard configuration, this array is of shape (16, 32, 32),
            with semantics (hist_idx, patch_row, patch_col).
    """
    coords_rel_to_hist = np.swapaxes(coords_rel_to_hist, 0, 1)
    xs, ys = np.abs(coords_rel_to_hist)
    y_contrib = 1 - (ys / (1/2 * const.descriptor_locality))
    x_contrib = 1 - (xs / (1/2 * const.descriptor_locality))
    contrib = y_contrib * x_contrib
    magnitude_path = magnitude_path * contrib
    return magnitude_path


def interpolate_1d_hist_contribution(magnitude_path: np.ndarray,
                                     orientation_patch: np.ndarray) -> np.ndarray:
    """ Interpolates an orientation's contribution between two orientation bins.
        When creating an orientation histogram, rather than adding an orientation's
        contribution to a single bin, it contributes mass to 2 bins, its left and
        right neighbor. This contribution is linear interpolated given the distance
        to each of these bins.

    Args:
        magnitude_path: The gradient magnitudes in the descriptor gradient patch.
        orientation_patch: The gradient orientations in the descriptor gradient patch.
    Returns:
        interpol_hist: The orientation histogram where contributions have been
            interpolated between neighboring bins.
    """
    nr_hists = magnitude_path.shape[0]
    orientation_patch = np.repeat(orientation_patch[None, ...], nr_hists, axis=0)

    hist_bin_width = const.descriptor_bin_width
    dist_to_next_bin = (orientation_patch % hist_bin_width)
    norm_dist_to_next_bin = dist_to_next_bin / hist_bin_width
    norm_dist_current_bin = 1 - norm_dist_to_next_bin

    current_bin_orients = orientation_patch
    next_bin_orients = (orientation_patch + hist_bin_width) % (2 * np.pi)

    hist_current = histogram_per_row(current_bin_orients.reshape((nr_hists, -1)),
                                     bins=const.nr_descriptor_bins,
                                     range_=(0, 2 * np.pi),
                                     weights=norm_dist_current_bin * magnitude_path)

    hist_next = histogram_per_row(next_bin_orients.reshape((nr_hists, -1)),
                                  bins=const.nr_descriptor_bins,
                                  range_=(0, 2 * np.pi),
                                  weights=norm_dist_to_next_bin * magnitude_path)

    interpol_hist = hist_current + hist_next
    return interpol_hist


def histogram_per_row(data: np.ndarray,
                      bins: int,
                      range_: tuple,
                      weights: np.ndarray) -> np.ndarray:
    """ Calculates histograms for each row of a 2D matrix.
        Has a similar signature to np.histogram(), except np.histogram() only
        supports 1D arrays. This function was created to speed up histogram
        creation for all (16) histograms in the descriptor patch. Borrows from
        https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis

    Args:
        data: A 2 dimensional array. A histogram will be calculated for each row.
        bins: The number of bins in the histograms.
        range_: The range of values that the histogram covers.
        weights: Contribution weights for each of the elements in `data`.
            This array must have the same number of elements as `data`.
    Returns:
        histograms: The histograms for each row. Represented as bin counts.
    """
    range_min, range_max = range_
    n_rows, n_cols = data.shape
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    idx = np.searchsorted(bin_edges, data, 'right') - 1
    bad_mask = idx == bins
    idx[bad_mask] = bins - 1
    scaled_idx = idx + bins * np.arange(n_rows)[:, None]
    limit = bins * n_rows
    histograms = np.bincount(scaled_idx.ravel(), minlength=limit, weights=weights.ravel())
    histograms.shape = (n_rows, bins)
    return histograms


def normalize_sift_feature(hists: np.ndarray) -> np.ndarray:
    """ Normalizes a keypoint's descriptor histograms to a unit length vector.
        See AOS section 4.2 and Lowe section 6.1

    Args:
        hists: A 1D array of a keypoint's descriptor histograms concatenated.
    Returns:
        hists: The histogram array that has been clipped and normalized to unit length.
    """
    hists = hists / np.linalg.norm(hists)
    hists = np.clip(hists, a_min=None, a_max=const.descriptor_clip_max)
    hists = hists / np.linalg.norm(hists)
    return hists


def assign_descriptor(keypoints: list[Keypoint],
                      gauss_octave: np.array,
                      octave_idx: int) -> list[Keypoint]:
    """ Assigns a descriptor to each keypoint.
        A descriptor is a collection of histograms that capture
        the distribution of gradients orientations in an oriented
        keypoint's local neighborhood. See AOS section 4.2 and Lowe
        section 6. Descriptors are created by taking a square
        patch of gradients surrounding the keypoint, assigning
        each gradient in the patch a coordinates relative to the
        oriented keypoint, and accumulating the gradients into a set
        of histograms. A gradient's contributions to a particular
        histogram is determined by its distance from the histogram's
        and keypoint's location.

    Args:
        keypoints: A list of keypoints that have been assigned an orientation.
        gauss_octave: An octave of Gaussian convolved images.
        octave_idx: The index of an octave.
    Returns:
        described_keypoints: A list of keypoints that have been assigned a descriptor.
    """
    magnitudes, orientations = gradients(gauss_octave)

    described_keypoints = list()
    for keypoint in keypoints:
        coord = keypoint.coordinate
        sigma = keypoint.sigma
        shape = gauss_octave.shape
        s, y, x = coord.round().astype(int)

        pixel_dist = pixel_dist_in_octave(octave_idx)
        max_width = (np.sqrt(2) * const.descriptor_locality * sigma) / pixel_dist
        max_width = max_width.round().astype(int)

        if patch_in_frame(coord, max_width, shape):
            orientation_patch = orientations[s,
                                             y - max_width: y + max_width,
                                             x - max_width: x + max_width]
            magnitude_patch = magnitudes[s,
                                         y - max_width: y + max_width,
                                         x - max_width: x + max_width]
            patch_shape = magnitude_patch.shape
            center_offset = [coord[1] - y, coord[2] - x]
            rel_patch_coords = relative_patch_coordinates(center_offset, patch_shape, pixel_dist, sigma,
                                                          keypoint.orientation)
            magnitude_patch = mask_outliers(magnitude_patch, rel_patch_coords, const.descriptor_locality)
            orientation_patch = (orientation_patch - keypoint.orientation) % (2 * np.pi)
            weights = weighting_matrix(center_offset, patch_shape, octave_idx, sigma, const.descriptor_locality)
            magnitude_patch = magnitude_patch * weights
            coords_rel_to_hists = rel_patch_coords[None] - histogram_centers[..., None, None]
            hists_magnitude_patch = mask_outliers(magnitude_patch[None], coords_rel_to_hists, const.inter_hist_dist, 1)
            hists_magnitude_patch = interpolate_2d_grid_contribution(hists_magnitude_patch, coords_rel_to_hists)
            hists = interpolate_1d_hist_contribution(hists_magnitude_patch, orientation_patch).ravel()
            keypoint.descriptor = normalize_sift_feature(hists)
            described_keypoints.append(keypoint)

    return described_keypoints
