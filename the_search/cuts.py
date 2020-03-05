#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cuts for testing dSph candidacy.

Author: Jack Runburg
Date: 22-08-2019 14:31


"""
import numpy as np
try:
    from utils import inverse_azimuthal_equidistant_coordinates
except ModuleNotFoundError:
    from .utils import inverse_azimuthal_equidistant_coordinates


def cut_on_pm(table, cut=5):
    """Return table with pm below threshold."""
    table = table[[cut >= np.hypot(pmra, pmdec) for pmra, pmdec in zip(table['pmra'].data, table['pmdec'].data)]]

    return table


def cut_on_parallax(table, cut):
    """Return objects with parallax magnitude below cut."""
    # Check if parallax is consistent with zero, indicating faraway objects
    if cut is None:
        return table

    indices = [(par - par_error * cut) < 0 for par, par_error in zip(table['parallax'].data, table['parallax_error'].data)]
    # indices = np.nonzero(indices)[0]
    return table[indices]


# PENDING
def parallax_test(dwarf, all_tables=True, table_index=-1, cuts=None, **kwargs):
    """Decide if dwarf passes the parallax test."""
    if all_tables is True:
        tables = dwarf.gaia_data
    else:
        tables = [dwarf.gaia_data[table_index]]

    if cuts is None:
        cuts = [10, 1]

    for table in tables:
        radius = table[0]

        pars = []
        for cut in cuts:
            tablex = cut_on_parallax(table[1], cut)
            parallax = tablex['parallax']
            pars.append(len(parallax))
        print(f'{dwarf.name} @ {radius}: {pars}, {pars[-1]/pars[0]}')


def proper_motion_test(dwarf, radius=0.5, cut=None, print_to_stdout=False, test_area=12, test_percentage=0.3, num_maxima=10, **kwargs):
    """Decide if dwarf passes proper motion test."""
    # message to print to log
    log_message = f': Proper motion test: area {test_area*2} x {test_area*2}, threshold {test_percentage*100}%, '

    # choose bins for histogram
    bound = 5
    bins = np.linspace(-bound, bound, num=20 * bound)

    # produce the histogram5
    if radius not in dwarf.gaia_data:
        dwarf.add_gaia_data(radius)

    table = cut_on_parallax(dwarf.gaia_data[radius], cut)
    histo, *_ = np.histogram2d(table['pmra'].data, table['pmdec'].data, bins=(bins, bins))

    # get indices of maxima of histo (end of array)
    histo_sorted = np.argsort(histo.flatten())
    histo_len = len(histo)
    total = np.sum(histo)

    # look at num_maxima amount of maxima
    subtotal = 0
    for index in range(-num_maxima, 0):
        # get coordinates of the maximum
        max_index = histo_sorted[index]
        coords = (max_index // histo_len, max_index % histo_len)

        # calculate the total counts in that area
        subtotal = np.sum(histo[coords[0] - test_area:coords[0] + test_area + 1, coords[1] - test_area:coords[1] + test_area + 1])

        # if above threshold, the test is a PASS
        if subtotal / total >= test_percentage:
            if print_to_stdout:
                print(f'{dwarf.name}, proper motion test: PASS')
            dwarf.log.append('PASS' + log_message + f'percentage {subtotal}')
            dwarf.tests.append(True)
            return True

    # if none of the num_maxima maxima are above the threshold, the test is a FAIL
    if print_to_stdout:
        print(f'{dwarf.name}, proper motion test: FAIL')
    dwarf.log.append('FAIL' + log_message + f'percentage {subtotal}')
    dwarf.tests.append(False)
    return False


def angular_density_test(dwarf, radii=None, print_to_stdout=False, density_tolerance=1.2, **kwargs):
    """Test for candidacy based on change in object density with changing angular size."""
    log_message = f': Angular density test: radii {radii}, tolerance {density_tolerance}, '

    # list for density values
    densities = []

    if radii is None:
        radii = [1.5, 0.1]
        # radii = [1, 0.1]
    # ensure there is gaia data for each radius
    for radius in radii:
        if radius not in dwarf.gaia_data:
            dwarf.add_gaia_data(radius)

        # get the density of objects for the given radius
        table = dwarf.gaia_data[radius]
        densities.append(len(table) / (radius**2))

    log_message += f'ratio {densities[-1] / densities[0]}'

    if densities[-1] / densities[0] > density_tolerance:
        if print_to_stdout:
            print(f'{dwarf.name}, density test: \tPASS')
        dwarf.log.append('PASS' + log_message)
        dwarf.tests.append(True)
        return True

    if print_to_stdout:
        print(f'{dwarf.name}, density test: \tFAIL')
    dwarf.log.append('FAIL' + log_message)
    dwarf.tests.append(False)
    return False

    # print(f'{dwarf.name}\t density ratio  {densities[-1]/densities[0]}\t {densities[-1]/densities[-2]}\t{densities[-2]/densities[0]}')


def poisson_overdensity_test(dwarf, table, table_radius, sigma=1.5, print_to_stdout=False):
    """Look for overdensities in cone using Poisson statistics."""
    # Get object counts for all available tables
    test_lengths = {radius: len(dwa_table) for (radius, dwa_table) in dwarf.gaia_data.items()}
    test_lengths[table_radius] = len(table)

    # Test each pair of radii (when one is smaller than the other)
    test_cases = [(radius1, radius2) for radius1 in [table_radius, *dwarf.gaia_data.keys()] for radius2 in [table_radius, *dwarf.gaia_data.keys()] if radius1 < radius2]

    # For each pair of radii, look for a sigma discrepancy of poisson statistics
    for (radius1, radius2) in test_cases:
        # mean is standard deviation
        poisson_sd = np.floor(test_lengths[radius2] * radius1**2/radius2**2)
        # print(f"{test_lengths[radius1]} compared to {sigma * poisson_sd} for {(radius1, radius2)}")
        if test_lengths[radius1] > sigma*poisson_sd:
            log_message = f"success at {radius1, radius2} with sd {poisson_sd} and result {test_lengths[radius1]-poisson_sd}"
            if print_to_stdout is True:
                print(f'{dwarf.name}, poisson density test: \tPASS')
                print(log_message)
            dwarf.log.append('PASS' + log_message)
            dwarf.tests.append(True)
            return True

        # if print_to_stdout is True:
        #     print(f'{dwarf.name}, poisson density test: \tFAIL')
        # dwarf.log.append('FAIL' + log_message)
    dwarf.tests.append(False)
    return False


def histogram_overdensity_test(convolved_data, histo_shape, region_ra, region_dec, outfile, mask, num_sigma=2, repetition=2):
    """Return coordinates with overdensities for all convolutions."""
    # Create zero array to search for overdensities
    passing = np.zeros(histo_shape)

    # For every radius probed, calculate the mean and sd of the histogram bins.
    for radius, convolved_array in convolved_data:
        hist_data, bins = np.histogram(convolved_array.flatten()[mask.flatten()], density=False, bins=101)
        midpoints = 0.5*(bins[1:] + bins[:-1])
        try:
            mean = np.average(midpoints, weights=hist_data)
        except ZeroDivisionError:
            print("Error with normalization")
            print(f"Potential boundary issue at ({region_ra}, {region_dec})")
            print(hist_data)
            mean = 0
        sd = np.sqrt(np.average((midpoints - mean)**2, weights=hist_data))

        # Add the overdensities to the test array
        passing += np.less(mean + num_sigma * sd, convolved_array)
        unique, counts = np.unique(passing, return_counts=True)
        # print(f"radius is {radius} with counts {counts}")

    # Passing candidates occur {repetition} times in the convolved data
    passing_indices_x, passing_indices_y = np.argwhere(passing > repetition).T

    return passing_indices_x, passing_indices_y


def pm_overdensity_test(convolved_data, histo_shape, region_ra, region_dec, outfile, mask, num_sigma=2, repetition=2):
    """Return coordinates with overdensities for all convolutions."""
    # Create zero array to search for overdensities
    passing = np.zeros(histo_shape)

    # For every radius probed, calculate the mean and sd of the histogram bins.
    for radius, convolved_array in convolved_data:
        hist_data, bins = np.histogram(convolved_array.flatten()[mask.flatten()], density=False, bins=101)
        midpoints = 0.5*(bins[1:] + bins[:-1])
        try:
            mean = np.average(midpoints, weights=hist_data)
        except ZeroDivisionError:
            print("Error with normalization")
            print(hist_data)
            mean = 0
        sd = np.sqrt(np.average((midpoints - mean)**2, weights=hist_data))

        # Add the overdensities to the test array
        passing += np.less(mean + num_sigma * sd, convolved_array)
        unique, counts = np.unique(passing, return_counts=True)
        # print(f"radius is {radius} with counts {counts}")

    # Passing candidates occur {repetition} times in the convolved data
    passing_indices_x, passing_indices_y = np.argwhere(passing > repetition).T

    if len(passing_indices_x) > 0:
        return True

    return False


############ WORKING
def overdensity_4d_test(convolved_data, histo_data, region_ra, region_dec, outfile, num_sigma=2, repetition=0, bins=101):
    """Return coordinates with overdensities in 4d pm-radec space for all convolutions."""
    X, Y, histo = histo_data
    passing = np.zeros(convolved_data[0][1].shape)
    min_radius = convolved_data[-1][0]

    for radius, convolved_array in convolved_data:
        hist_data, bins = np.histogram(convolved_array.flatten(), density=False, bins=bins)
        # print(hist_data)
        # print("bins", bins)
        hist_data.dtype = np.float64
        midpoints = 0.5*(bins[2:] + bins[1:-1])
        mean = np.average(midpoints, weights=hist_data[1:])
        sd = np.sqrt(np.average((midpoints - mean)**2, weights=hist_data[1:]))
        print('for radius ', radius, " the value is ", mean, " +/- ", sd)
        # print("sd", sd)

        # print("passing ", passing.shape)
        # print("convolved array ", convolved_array.shape)
        passing += np.less(mean + num_sigma * sd, convolved_array)
        # print(np.amax(hist_data[1:]))
        # print(np.amax(passing))
        # unique, counts = np.unique(passing, return_counts=True)
        # print(f"radius is {radius} with counts {counts}")

    passing_indices_x, passing_indices_y, *passing_pm = np.argwhere(passing > repetition).T
    print(len(passing.flatten()), len(passing_indices_x))
    passing_ra, passing_dec = inverse_azimuthal_equidistant_coordinates(np.deg2rad(X[passing_indices_x] + min_radius/2), np.deg2rad(Y[passing_indices_y]+min_radius/2), region_ra, region_dec)

    with open(outfile, 'w') as outfl:
        for ra, dec in zip(passing_ra, passing_dec):
            outfl.write(f"{ra} {dec}\n")

    print("output to ", outfile)

    return X[passing_indices_x] + min_radius/2, Y[passing_indices_y]+min_radius/2
