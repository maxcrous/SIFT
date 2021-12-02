"""
This file contains functions related to finding candidate keypoints
for SIFT features in an image. The central function in this file is
    `find_keypoints`
"""

from typing import Tuple

import numpy as np

import const
from octaves import absolute_sigma, derivatives, pixel_dist_in_octave


class Keypoint:
    """ A keypoint object that is created when a reference orientation is found.
        A descriptor is assigned to the keypoint at a later time.
    """
    def __init__(self, coord, octave_idx, orientation):
        self.octave_idx: int = octave_idx
        self.coordinate: np.ndarray = coord
        self.orientation: float = orientation
        self.sigma: float = absolute_sigma(octave_idx=self.octave_idx, layer_idx=coord[0])
        self.descriptor = None

    @property
    def absolute_coordinate(self):
        """ Calculates the keypoint's coordinates relative to the input image. """
        s, y, x = self.coordinate
        pixel_dist = pixel_dist_in_octave(self.octave_idx)
        return np.array([self.sigma, y * pixel_dist, x * pixel_dist])


def in_frame(coord: np.ndarray, shape: np.ndarray) -> bool:
    """ Determines whether a coordinate falls within an array.

    Args:
        coord: A coordinate.
        shape: The shape of the an array.
    Returns :
        True if in frame, False if not.
    """
    return (coord >= 0).all() and (coord <= shape - 1).all()


def interpolation_offsets(deriv: np.ndarray,
                          second_deriv: np.ndarray) -> Tuple[np.ndarray, float]:
    """ Calculates the coordinate offset and value offset of an extremum,
        relative to the non-interpolated extremum.

    Args:
        deriv: The extremum's first derivatives.
        second_deriv: The extremum's second derivatives.
    Returns:
        offset: The extremum's offset from the non-interpolated coordinate.
        val_change: The extremum's offset from the non-interpolated value.
    """
    second_deriv = second_deriv.reshape((3, 3))
    second_deriv_inv = np.linalg.pinv(second_deriv)
    offset = -np.dot(second_deriv_inv, deriv)
    val_change = (1 / 2) * np.dot(deriv, offset)
    return offset, val_change


def pass_edge_test(second_deriv: np.ndarray) -> bool:
    """ Eliminates keypoints along edges for improved detection
        repeatability. Returns true when a keypoint passes edge
        test, i.e., when the keypoint does *not* lie on an edge.
        For details see Lowe section 4.1 and AOS section 3.3.

    Args:
        second_deriv: Hessian of the keypoint.
    Returns:
        passes: Whether the keypoint passes the edge test.
    """
    xy_hessian = second_deriv.reshape((3, 3))[1:, 1:].copy()
    trace = np.trace(xy_hessian)
    det = np.linalg.det(xy_hessian) + const.eps
    threshold = ((const.edge_ratio_thresh + 1) ** 2) / const.edge_ratio_thresh
    return (trace ** 2) / (det + const.eps) < threshold


def interpolate(extremum_coord: np.ndarray,
                dog_octave: np.ndarray,
                derivs: np.ndarray,
                second_derivs: np.ndarray) -> Tuple[bool, np.ndarray, float]:
    """ Interpolates the coordinate and value of an extremum in
        a Difference of Gaussian octave. Interpolation is performed
        by a Taylor expansion to fit a 3D quadratic function to the
        local sample. The interpolated extremum is where this function's
        derivative equals 0. This enables more precise sub-pixel keypoint
        locations, which improves descriptor quality. For details,
        see Lowe section 4 and AOS section 3.2.

    Args:
        extremum_coord: Non-interpolated coordinates of an extremum.
        dog_octave: An octave of Difference of Gaussian images.
        derivs: The s, x, and y derivatives of the octave.
        second_derivs: The ss, yy, xx, sy, sx, yx derivatives of the octave.
            Provided in the format of a flattened Hessian. Indexing is as
            follows: second_derivs[:, s, y, x] is the hessian at point [s, y, x].
    Returns:
        success: Whether interpolation was successful.
        interpol_coord: The interpolated extremum coordinate.
        interpol_val: The interpolated extremum magnitude.
    """
    interpol_coord = extremum_coord
    interpol_val = dog_octave[tuple(interpol_coord)]
    shape = np.array(dog_octave.shape)
    success = False

    for _ in range(const.max_interpolations):
        s, y, x = interpol_coord.round().astype(int)
        deriv = derivs[:, s, y, x]
        second_deriv = second_derivs[:, s, y, x]
        offset, val_change = interpolation_offsets(deriv, second_deriv)
        interpol_coord = interpol_coord + offset
        interpol_val = interpol_val + val_change

        if (abs(offset) < const.offset_thresh).all() and in_frame(interpol_coord, shape):
            success = True
            break

        elif not in_frame(interpol_coord, shape):
            break

    return success, interpol_coord, interpol_val


def find_keypoints(extrema: np.ndarray, dog_octave: np.ndarray) -> np.ndarray:
    """ Finds valid keypoint coordinates among candidate Difference of
        Gaussian extrema. A candidate keypoint coordinate must be
        interpolated, pass a magnitude threshold, and pass the
        edge test.

    Args:
        extrema: Difference of Gaussian extrema coordinates.
        dog_octave: An octave of Difference of Gaussian images.
    Returns:
        keypoint_coords: Interpolated coordinates of potential keypoints.
    """
    keypoint_coords = list()
    derivs, second_derivs = derivatives(dog_octave)

    for extremum_coord in extrema.T:
        if abs(dog_octave[tuple(extremum_coord)]) > const.coarse_magnitude_thresh:
            success, extremum_coord, extremum_val = interpolate(extremum_coord, dog_octave, derivs, second_derivs)
            if success and abs(extremum_val) > const.magnitude_thresh:
                s, y, x = extremum_coord.round().astype(int)
                if pass_edge_test(second_derivs[:, s, y, x]):
                    keypoint_coords.append(extremum_coord)

    return np.array(keypoint_coords)
