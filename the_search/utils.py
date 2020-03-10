#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Support functions for the search.

Author: Jack Runburg
Date: 22-08-2019 14:30


"""

import warnings
from random import random
from time import sleep
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import numpy as np


def convolve_spatial_histo(gaia_table, region_radius, radii):
    """Convolve the spatial histogram of GAIA data with bin sizes given in radii.


    Inputs:
        - gaia_table: table of gaia data
        - region_radius: radius of region in degrees
        - radii: list of radii in degrees
    """
    from astropy import convolution

    # Bin data at finest resolution
    min_radius = min(radii)
    histo, xedges, yedges = np.histogram2d(gaia_table['x'], gaia_table['y'], bins=region_radius//min_radius)
    # print(histo.shape)
    # put bins in degrees
    xedges *= 180/np.pi
    yedges *= 180/np.pi

    # Set bins for plotting
    X, Y = np.meshgrid(xedges, yedges)
    histo_mask = np.less(X[:-1, :-1]**2 + Y[:-1, :-1]**2, region_radius**2)

    # Convolve the histogram with different size tophats
    convolved_data = []
    for radius in radii:
        convolution_kernel = convolution.Tophat2DKernel(radius//min_radius)
        histo_mask = np.less(X[:-1, :-1]**2 + Y[:-1, :-1]**2, region_radius**2)
        convolved_array = np.multiply(convolution.convolve(histo, convolution_kernel), histo_mask)

        # All convolved data is stored here
        convolved_data.append((radius, convolved_array))

        print(f"finished {radius}")

    return convolved_data, xedges, yedges, X, Y, histo, histo_mask


def convolve_pm_histo(gaia_table, region_radius, radii):
    """Convolve the pm histogram of GAIA data with bin sizes given in radii.


    Inputs:
        - gaia_table: table of gaia data
        - region_radius: radius of region in degrees
        - radii: list of radii in degrees
    """
    from astropy import convolution

    # Bin data at finest resolution
    min_radius = min(radii)
    histo, xedges, yedges = np.histogram2d(gaia_table['pmra'], gaia_table['pmdec'], bins=5//min_radius)

    # Set bins for plotting
    X, Y = np.meshgrid(xedges, yedges)
    histo_mask = np.less(X[:-1, :-1]**2 + Y[:-1, :-1]**2, region_radius**2)

    # Convolve the histogram with different size tophats
    convolved_data = []
    for radius in radii:
        convolution_kernel = convolution.Tophat2DKernel(radius//min_radius)
        histo_mask = np.less(X[:-1, :-1]**2 + Y[:-1, :-1]**2, region_radius**2)
        convolved_array = np.multiply(convolution.convolve(histo, convolution_kernel), histo_mask)

        # All convolved data is stored here
        convolved_data.append((radius, convolved_array))

        print(f"finished {radius}")

    return convolved_data, xedges, yedges, X, Y, histo, histo_mask


def gaia_region_search(ra, dec, outfile, radius=10, sigma=3, pm_threshold=5, bp_rp_threshold=2, dump_to_file=True):
    """Given coordinates, search gaia around a region and populate cones within that region.


    Inputs:
        - ra: right ascension of region in degrees
        - dec: declination of region in degrees
        - outfile: file to dump results to
        - radius: radius of region (half of side length of a box) in degrees
        - sigma: number of sigma to check parallax consistency with zero (far away)
        - pm_threshold: maximum magnitude of proper motion in mas/yr
        - bp_rp_threshold: maximum value of bp_rp
        - dump_to_file: boolean to save file

    Return:
        - Asynchronous job query
    """
    warnings.filterwarnings("ignore", module='astropy.*')
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
    job = Gaia.launch_job_async(f"SELECT TOP 10000000 \
                                gaia_source.source_id,gaia_source.ra,gaia_source.ra_error,gaia_source.dec, \
                                gaia_source.dec_error,gaia_source.parallax,gaia_source.parallax_error, \
                                gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error, \
                                gaia_source.bp_rp, gaia_source.phot_g_mean_mag \
                                FROM gaiadr2.gaia_source \
                                WHERE \
                                CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),BOX('ICRS',{coords.ra.degree},{coords.dec.degree},{2*radius},{2*radius}))=1 AND  (gaiadr2.gaia_source.parallax - gaiadr2.gaia_source.parallax_error * {sigma} <= 0) AND (SQRT(POWER(gaiadr2.gaia_source.pmra, 2) + POWER(gaiadr2.gaia_source.pmdec, 2)) <= {pm_threshold}) AND (gaiadr2.gaia_source.bp_rp <= {bp_rp_threshold})", dump_to_file=dump_to_file, output_file=outfile, verbose=True)

    return job


def azimuthal_equidistant_coordinates(gaia_table, region_ra, region_dec):
    """Return cartesian coordinates from GAIA table using azimuthal equidistant projection.

    Project the sphere onto a 2D surface without distorting the regions near the center of projection too much. 


    Inputs:
        - gaia_table: data_table from gaia
        - region_ra: right ascension of region in degrees
        = region_dec: declination of region in degrees

    Returns:
        - x, y the projections of ra, dec"""
    # Use the notation given here:
    # http://mathworld.wolfram.com/AzimuthalEquidistantProjection.html

    ra_rad = np.deg2rad(region_ra)
    dec_rad = np.deg2rad(region_dec)

    ra_gaia_rad = np.deg2rad(gaia_table['ra'])
    dec_gaia_rad = np.deg2rad(gaia_table['dec'])

    c = np.arccos(np.sin(dec_rad)*np.sin(dec_gaia_rad) + np.cos(dec_rad)*np.cos(dec_gaia_rad)*np.cos(ra_gaia_rad-ra_rad))

    k_prime = c / np.sin(c)

    x = k_prime * np.cos(dec_gaia_rad) * np.sin(ra_gaia_rad - ra_rad)
    y = k_prime * (np.cos(dec_rad)*np.sin(dec_gaia_rad) - np.sin(dec_rad)*np.cos(dec_gaia_rad)*np.cos(ra_gaia_rad-ra_rad))

    return x, y


def inverse_azimuthal_equidistant_coordinates(x, y, ra_rad, dec_rad):
    """Given (x, y) positions from AEP, return (ra, dec) in deg.


    Inputs:
        - x: projected cartesian coordinates
        - y: projected cartesian coordinates
        - ra_rad: region ra in radians
        - dec_rad: region dec in radians

    Returns:
        - ra, dec in degrees"""
    # http://mathworld.wolfram.com/AzimuthalEquidistantProjection.html
    c = np.sqrt(x**2 + y**2)

    phi = np.arcsin(np.cos(c)*np.sin(dec_rad) + y/c * np.sin(c)*np.cos(dec_rad))
    if dec_rad == np.pi/2:
        lamb = ra_rad + np.arctan2(-x, y)
    elif dec_rad == -np.pi/2:
        lamb = ra_rad + np.arctan2(x, y)
    else:
        lamb = ra_rad + np.arctan2(x*np.sin(c), (c*np.cos(dec_rad)*np.cos(c) - y*np.sin(dec_rad)*np.sin(c)))

    return np.rad2deg(lamb), np.rad2deg(phi)


def generate_full_sky_cones(cone_radius, galactic_plane=15, hemi='north', out_to_file=True, output_directory='./region_list/'):
    """Generate full sky coverage of candidate cones.
    
    Distribute points over the full sky and write them to a file.


    Inputs:
        - cone_radius: radius of the cone in degrees
        - galactic_plane: galactic latitude of the galactic plane (i.e. the symmetric band about 0 to exclude) in degrees
        - hemi: hemisphere to consider [values are 'both', 'north', 'south']
        - out_to_file: boolean whether to save to file
        - output_directory: path to save file

    Returns:
        - ra, dec of cones if not output to file
    """
    angle = 90 - galactic_plane
    deg_values = np.arange(-angle, angle, cone_radius)
    x, y = np.meshgrid(deg_values, deg_values)

    x = np.concatenate((x.flatten(), (x + cone_radius/2)[:, :-1].flatten()))
    y = np.concatenate((y.flatten(), (y + cone_radius/2)[:-1, :].flatten()))

    inside_of_circle = np.less(x**2+y**2, angle**2)
    x = x[inside_of_circle]
    y = y[inside_of_circle]

    # NORTHERN HEMISPHERE
    ra, dec, ra2, dec2 = [], [], [], []
    if hemi == 'north' or hemi == 'both':
        ra, dec = inverse_azimuthal_equidistant_coordinates(np.deg2rad(x), np.deg2rad(y), 0, np.pi/2)
    # SOUTHERN HEMISPHERE
    if hemi == 'south' or hemi == 'both':
        ra2, dec2 = inverse_azimuthal_equidistant_coordinates(np.deg2rad(x), np.deg2rad(y), 0, -np.pi/2)

    ra_gal = np.concatenate((ra, ra2))
    dec_gal = np.concatenate((dec, dec2))

    ra, dec = galactic_to_icrs(ra_gal, dec_gal)

    if out_to_file is True:
        candidate_per_file = 250
        for i in range(1, len(ra)//candidate_per_file+1):
            with open(output_directory + f"region{i}.txt", 'w') as outfile:
                outfile.write("# candidates for full sky search of dsph\n")
                np.savetxt(outfile, np.array([ra[candidate_per_file*(i-1):candidate_per_file*i], dec[candidate_per_file*(i-1):candidate_per_file*i]]).T, delimiter=" ", comments='#')
    else:
        return ra, dec


def galactic_to_icrs(ra_gal, dec_gal):
    """Return galactic coordinates."""
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(ra_gal, dec_gal, unit='deg', frame='galactic')
    return coords.icrs.ra, coords.icrs.dec


def icrs_to_galactic(ra_icrs, dec_icrs):
    """Return icrs coordinates."""
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(ra_icrs, dec_icrs, unit='deg', frame='icrs')
    return coords.galactic.l, coords.galactic.b


def cut_out_candidates_close_to_plane_and_slmc(ra, dec, latitude=20, output=True, far_file=None, near_file=None):
    """Reduce candidate list by separating out regions near the galactic plane.

    
    Inputs:
        - ra: list of ra of candidates
        - dec: list of dec of candidates
        - latitude: minimum absolut value of galactic latitude (the region to cut out)
        - output: boolean whether to dump filtered candidates to file
        - far_file: file for candidates outside of excluded region
        - near_file: file for candidates near excluded region
    """
    l_gal, b_gal = icrs_to_galactic(ra, dec)

    indices_too_close_to_plane = np.logical_and(np.less(b_gal.value, latitude), np.less(-latitude, b_gal.value))

    indices_too_close_to_mag_cloud = np.logical_or(np.less(np.rad2deg(angular_distance(ra, dec, 80.89, -69.76)), 13), np.less(np.rad2deg(angular_distance(ra, dec, 13.16, -72.8)), 9))

    near_indices = np.logical_or(indices_too_close_to_plane, indices_too_close_to_mag_cloud)

    ra_far, dec_far = ra[~near_indices], dec[~near_indices]
    ra_close, dec_close = ra[near_indices], dec[near_indices]

    if output is True:
        with open(far_file, 'w') as outfile:
            np.savetxt(outfile, np.array([ra_far, dec_far]).T, delimiter=" ")

        with open(near_file, 'w') as outfile:
            np.savetxt(outfile, np.array([ra_close, dec_close]).T, delimiter=" ")

    return ra_far, dec_far, ra_close, dec_close


def fibonnaci_sphere(num_points, limit=16, point_start=0, point_end=None):
    """Return a coordinate on a Fibonnaci sphere."""
    if point_end is None:
        point_end = num_points
    for point in range(int(point_start), int(point_end)):
        if point % 100 == 0:
            print(point)
        point += 0.5

        # equally spaced coordinates
        theta = 180/np.pi * (np.arccos(1 - 2 * point / num_points) - np.pi/2)
        phi = 180 * (1 + 5**0.5) * point

        if abs(theta) > limit:
            c_gal = SkyCoord(phi, theta, unit='deg', frame='galactic')
            icrs_coords = (c_gal.icrs.ra.value, c_gal.icrs.dec.value)

            # lmc_coords = SkyCoord(80.89, -69.76, unit='deg')
            # lmc = CircleSkyRegion(lmc_coords, Angle(5.5, 'deg'))
            # if icrs_coords not in lmc:
            yield icrs_coords


if __name__ == '__main__':
    # gaia_region_search(90, 90)
    # for _ in range(20):
    #     print('{}, {}'.format(*random_cones_outside_galactic_plane()))
    # get_cone_in_region(10, 20, 5, num_cones=20)
    # print(inverse_azimuthal_equidistant_coordinates(np.array([0]), np.array([0.0001]), 0.001, -np.pi/2))
    import matplotlib.pyplot as plt
    from astropy import units as u
    import astropy.coordinates as coord

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection="hammer")
    ra, dec = generate_full_sky_cones(3.16, out_to_file=False, hemi='both')
    ra = coord.Angle(ra)
    ra = ra.wrap_at(180*u.deg)
    dec = coord.Angle(dec)
    ax.scatter(ra.radian, dec.radian, color='xkcd:light grey blue')
    fig.savefig("allsampleconesplot.pdf")
