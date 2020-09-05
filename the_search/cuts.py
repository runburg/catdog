#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cuts for testing dSph candidacy.

Author: Jack Runburg
Date: 22-08-2019 14:31


"""
import numpy as np
from scipy import stats
try:
    from utils import inverse_azimuthal_equidistant_coordinates
except ModuleNotFoundError:
    from .utils import inverse_azimuthal_equidistant_coordinates


def histogram_overdensity_test(convolved_data, histo_shape, region_ra, region_dec, region_rad, outfile, mask, num_sigma=2, repetition=2, threshold_prob=0.95):
    """Return coordinates with overdensities for all convolutions.

    Look for overdensities by looking for large bin counts above the average (background).


    Inputs:
        - convolved_data: list of tuples (radius, convolved_array) where radius is radius in degrees of convolution kernel and convolved array is the result
        - histo_shape: shape of histogrammed data
        - region_ra: right ascension in """
    # Create zero array to search for overdensities
    passing = np.zeros(histo_shape)

    # For every radius probed, calculate the mean and sd of the histogram bins.
    for radius, convolved_array in convolved_data:
        hist_data, bins = np.histogram(convolved_array.flatten()[(~np.isnan(convolved_array)).flatten()], density=False, bins=101, range=(0, np.nanmax(convolved_array)))
        # print(hist_data)
        midpoints = 0.5*(bins[1:] + bins[:-1])
        try:
            mean = np.average(midpoints, weights=hist_data)
        except ZeroDivisionError:
            print("Error with normalization")
            print(f"Potential boundary issue at ({region_ra}, {region_dec})")
            print(hist_data)
            return [], []
        sd = np.sqrt(np.average((midpoints - mean)**2, weights=hist_data))

        # Add the overdensities to the test array
        count_pass = stats.poisson.ppf(1 - (1 - threshold_prob) * radius**2 / region_rad**2, mean)
        passing += np.less(count_pass, convolved_array)
        unique, counts = np.unique(passing, return_counts=True)
        # print(f"radius is {radius} with counts {counts}")

    # Passing candidates occur {repetition} times in the convolved data
    passing_indices_x, passing_indices_y = np.argwhere(passing > repetition).T

    return passing_indices_x, passing_indices_y


def pm_xi2_test(pm, error_pm):
    """Calculate Xi^2 for pmra and pmdec."""
    mean = np.mean(pm)

    dof = len(pm) - 1
    chi2 = np.sum((pm - mean)**2 / error_pm**2)

    return chi2 / dof


def pm_overdensity_test(convolved_data, histo_shape, region_ra, region_dec, outfile, mask, num_sigma=2, repetition=2):
    """Return coordinates with overdensities for all convolutions."""
    # Create zero array to search for overdensities
    passing = np.zeros(histo_shape)

    # For every radius probed, calculate the mean and sd of the histogram bins.
    for radius, convolved_array in convolved_data:
        hist_data, bins = np.histogram(convolved_array.flatten()[~np.isnan(convolved_array).flatten()], density=False, bins=101, range=(0, np.nanmax(convolved_array)))
        midpoints = 0.5 * (bins[1:] + bins[:-1])
        try:
            mean = np.average(midpoints, weights=hist_data)
        except ZeroDivisionError:
            print("Error with normalization")
            continue

        sd = np.sqrt(np.average((midpoints - mean)**2, weights=hist_data))

        # Add the overdensities to the test array
        passing += np.less(mean + num_sigma * sd, convolved_array)
        unique, counts = np.unique(passing, return_counts=True)
        # print(f"radius is {radius} with counts {counts}")

    # Passing candidates occur {repetition} times in the convolved data
    passing_indices_x, passing_indices_y = np.argwhere(passing > repetition).T

    return passing_indices_x, passing_indices_y
