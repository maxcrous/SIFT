"""
This file contains the main function for demoing the SIFT implementation.

Throughout this repository,
mentions of 'AOS' refer to
    Anatomy of the SIFT Method. Otero, Ives Rey. Diss. École normale supérieure de Cachan-ENS Cachan, 2015.
mentions of 'Lowe' refer to
    Sift-the scale invariant feature transform. Lowe. Int. J, 2(91-110), 2, 2004.
"""

import cv2
import numpy as np

from descriptor import assign_descriptor
from keypoints import Keypoint, find_keypoints
from match import match_sift_features, visualize_matches
from octaves import build_gaussian_octaves, build_dog_octave, find_dog_extrema
from reference_orientation import assign_reference_orientations


def detect_sift_features(img: np.ndarray) -> list[Keypoint]:
    """ Detects SIFT keypoints in an image.

    Args:
        img: A grayscale image with values in range [0, 1].
    Returns:
        keypoints: A list of keypoints objects, each containing
            a coordinate, reference orientation and descriptor.
    """
    gauss_octaves = build_gaussian_octaves(img)

    features = list()
    for octave_idx, gauss_octave in enumerate(gauss_octaves):
        dog_octave = build_dog_octave(gauss_octave)
        extrema = find_dog_extrema(dog_octave)
        keypoint_coords = find_keypoints(extrema, dog_octave)
        keypoints = assign_reference_orientations(keypoint_coords, gauss_octave, octave_idx)
        keypoints = assign_descriptor(keypoints, gauss_octave, octave_idx)
        features += keypoints

    return features


def main():
    """ Detects and matches SIFT features in two images. """
    img1 = cv2.imread('images/box.png', flags=cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/box_in_scene.png', flags=cv2.IMREAD_GRAYSCALE)
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    keypoints1 = detect_sift_features(img1)
    keypoints2 = detect_sift_features(img2)
    matches = match_sift_features(keypoints1, keypoints2)
    visualize_matches(matches, img1, img2)
    print(len(matches))

    print('len of keypoints 1: ', len(keypoints1))
    print('len of keypoints 2: ', len(keypoints2))
    print('len of matches: ', len(matches))


if __name__ == '__main__':
    main()

