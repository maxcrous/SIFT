"""
This file contains functions related to matching and visualizing SIFT features.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

import const
from keypoints import Keypoint


def match_sift_features(features1: list[Keypoint],
                        features2: list[Keypoint]) -> list[tuple[Keypoint, Keypoint]]:
    """ A brute force method for finding matches between two sets of SIFT features.

    Args:
        features1: A set of SIFT features.
        features2: A set of SIFT features.
    Returns:
        matches: A list of matches. Each match is a (feature, feature) tuples.
    """

    matches = list()

    for idx1, feature1 in enumerate(features1):
        descriptor1 = feature1.descriptor

        min_dist = np.inf
        rest_min = np.inf
        min_feature = None

        for idx2, feature2 in enumerate(features2):
            descriptor2 = feature2.descriptor

            dist = np.linalg.norm(descriptor1 - descriptor2)

            if dist < min_dist:
                min_dist = dist
                min_feature = feature2

            elif dist < rest_min:
                rest_min = dist

        if min_dist < rest_min * const.rel_dist_match_thresh:
            matches.append((feature1, min_feature))

    return matches


def visualize_matches(matches: list[tuple[Keypoint, Keypoint]],
                      img1: np.ndarray,
                      img2: np.ndarray):
    """ Plots SIFT keypoint matches between two images.

    Args:
        matches: A list of matches. Each match is a (feature, feature) tuples.
        img1: The image in which the first match features were found.
        img2: The image in which the second match features were found.
    """

    coords_1 = [match[0].absolute_coordinate for match in matches]
    coords_1y = [coord[1] for coord in coords_1]
    coords_1x = [coord[2] for coord in coords_1]
    coords_1xy = [(x, y) for x, y in zip(coords_1x, coords_1y)]

    coords_2 = [match[1].absolute_coordinate for match in matches]
    coords_2y = [coord[1] for coord in coords_2]
    coords_2x = [coord[2] for coord in coords_2]
    coords_2xy = [(x, y) for x, y in zip(coords_2x, coords_2y)]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(img1, cmap='Greys_r')
    ax2.imshow(img2, cmap='Greys_r')

    ax1.scatter(coords_1x, coords_1y)
    ax2.scatter(coords_2x, coords_2y)

    for p1, p2 in zip(coords_1xy, coords_2xy):
        con = ConnectionPatch(xyA=p2, xyB=p1, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)

    plt.show()
