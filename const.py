"""
This file contains the SIFT algorithm's parameters.
When a docstring mentions 'the standard configuration of SIFT',
it refers to these parameters, as provided in the repository's main branch.
These parameters were chosen in accordance with the AOS paper.

Mentions of 'AOS' refer to
    Anatomy of the SIFT Method. Otero, Ives Rey. Diss. École normale supérieure de Cachan-ENS Cachan, 2015.
Mentions of 'Lowe' refer to
    Sift-the scale invariant feature transform. Lowe. Int. J, 2(91-110), 2, 2004.
"""

import math

import numpy as np

eps = np.finfo(dtype=float).eps


###################
# Octaves
###################

# The number of Gaussian and Difference of Gaussian octaves that are generated,
# Referred to as n_oct in AOS
nr_octaves = 8

# The number of scales generated per Gaussian octave.
# Referred to as n_spo in AOS
scales_per_octave = 3

# Auxiliary scales per octave. Required to generate enough scales in the Gaussian
# scale space such that the Difference of Gaussian layers span the desired scales.
# Here, desired scales means the doubling of the Gaussian blur's standard deviation
# after each DoG octave and having 3 DoG layers per octave in which extrema can be found.
# See AOS Figure 2.(b). and Figure 5.
auxiliary_scales = 3

# The assumed geometric distance between pixels in the original image.
# Referred to as delta_in in AOS.
orig_pixel_dist = 1

# The pixel distance in the first layer of the first octave,
# i.e., upscaled version of the input image.
# Referred to as delta_min in AOS.
min_pixel_dist = 0.5

# The amount by which the input image is upscaled for the first octave.
first_upscale = orig_pixel_dist / min_pixel_dist

# The assumed blur level of the input image
# Referred to as sigma_in in AOS.
orig_sigma = 0.5

# The desired blur level for the first octave's first layer.
# Referred to as sigma_min in AOS.
min_sigma = 0.8

# The sigma required to move from the input image to the first octave's first layer.
# See AOS section 2.2 formula 6.
init_sigma = 1 / min_pixel_dist * math.sqrt(min_sigma ** 2 - orig_sigma ** 2)


#########################
# Keypoint Tests
#########################

# Determines whether a Difference of Gaussian extrema is large enough.
# Referred to as C_dog in AOS. See AOS section 3.3.
magnitude_thresh = 0.015 * ((2 ** (1 / scales_per_octave) - 1) / (2 ** (1 / 3) - 1))

# Before attempting to interpolate a DoG extrema's value, it must have
# a magnitude larger than this threshold.
coarse_magnitude_thresh = 0.85 * magnitude_thresh

# Determines whether a coordinate offset of the interpolated extremum
# is too large. If it is too large, the interpolated offset must be recalculated relative
# to a different sampling point. This threshold is et to 0.5 in Lowe and 0.6 in AOS.
# See AOS section 3.2.
offset_thresh = 0.6

# Determines whether the the eigenvalue ratio of the hessian is too large.
# A eigenvalue ratio larger than this value means the keypoint must be discarded,
# as it probably lies on an edge or other poorly defined feature.
edge_ratio_thresh = 10

# Maximum number of attempts for interpolating an extrema
# See AOS section 3.4 algorithm 6.
max_interpolations = 3


#########################
# Reference orientation
#########################

# The number of bins in the reference orientation histogram.
# Referred to as n_bins in AOS. See AOS section 4.1.
nr_bins = 36

# Controls how "local" or "close" to the keypoint the reference orientation
# analysis is performed. For example, this value is used to set the size of the
# patch used for reference orientation finding, and for weighting the contribution
# of gradients in the local neighborhood to find this reference orientation.
# Increasing this value would result in a larger neighborhood being considered.
# Referred to as lambda_ori in AOS.
reference_locality = 1.5

# An additional constant to control the size of the reference orientation patch.
# See AOS Figure 7.
reference_patch_width_scalar = 6 * reference_locality

# Number of smoothing steps performed with a three-tap box filter ([1, 1, 1])
# on the reference orientation histogram. See AOS section 4.1.B.
nr_smooth_iter = 6

# The magnitude (relative to the histogram's largest peak)
# that an orientation bin must reach to be considered a reference
# orientation. Referred to as 't' in AOS. See blue line in AOS Figure 8.
rel_peak_thresh = 0.8

# The number of bins to the left and right of a peak that are ignored when searching
# for the next potential local maximum in the reference orientation histogram.
mask_neighbors = 4

# The maximum number of local maxima that will be found in a gradient orientation histogram.
max_orientations_per_keypoint = 2


#########################
# Descriptor
#########################

# Controls how "local" or "close" to the keypoint the descriptor analysis
# is performed. For example, this value is used to set the size of the
# patch used for descriptor finding, and for weighting the contribution
# of gradients in the descriptor. Increasing this value would result in
# a larger neighborhood being considered. Referred to as lambda_descr in AOS.
descriptor_locality = 6

# The number of rows in the square descriptor grid. Each cell in this grid
# has a histogram associated with it. Referred to as n_hist in AOS.
nr_descriptor_histograms = 4

# The distance between histogram centers along the x or y axis.
inter_hist_dist = descriptor_locality / nr_descriptor_histograms

# The number of orientation bins in a descriptor histogram.
# Referred to as n_ori in AOS.
nr_descriptor_bins = 8

# The width of an descriptor histogram's orientation bin.
descriptor_bin_width = nr_descriptor_bins / (2 * np.pi)

# The normalized maximum amount of mass that may be assigned to a single
# bin in the SIFT feature, i.e., the concatenated descriptor histograms.
# See AOS section 4.2. The SIFT feature vector.
descriptor_clip_max = 0.2


#########################
# Matching
#########################

# The distance factor between first and second nearest neighbor for accepting a feature match.
# I.e, create a match if:  first_nn_dist < second_nn_dist * `rel_dist_match_thresh`
rel_dist_match_thresh = 0.6
