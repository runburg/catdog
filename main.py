# -*- coding: utf-8 -*-
"""Main script for running the dSph search and controlling output.

Author: Jack Runburg
Date: 22-08-2019 14:03

"""

import warnings
import numpy as np
import sys
import os
from zipfile import ZipFile, ZIP_DEFLATED
from astropy.table import Table
from the_search import cuts
from the_search.utils import gaia_region_search, azimuthal_equidistant_coordinates, inverse_azimuthal_equidistant_coordinates, convolve_spatial_histo, convolve_pm_histo, outside_of_galactic_plane
from the_search.plots import convolved_histograms, convolved_histograms_1d, new_all_sky, convolved_pm_histograms

warnings.filterwarnings("ignore")


def filter_then_plot(infiles, prefix='./candidates/', gal_plane_setting=15, radius=3.16, outfile='all_sky_plot.pdf', dif_color_for_each_file=False, labs=[]):
    """Create all sky plot and filter candidates."""
    from the_search.utils import cut_out_candidates_close_to_plane_and_slmc

    region_rad = radius
    coord_list = []
    new_color_at = []
    for infile in infiles:
        coords = np.loadtxt(prefix + infile, delimiter=" ")
        coord_list.append(coords)
        new_color_at.append(len(coords))

    # print(new_color_at)
    coord_list = np.concatenate(coord_list)

    filtered_cand_file = prefix + "successful_candidates_filtered.txt"
    near_cand_file = prefix + "successful_candidates_near.txt"

    ra_suc, dec_suc, ra_near, dec_near, new_color_at = cut_out_candidates_close_to_plane_and_slmc(coord_list[:, 0], coord_list[:, 1], far_file=filtered_cand_file, near_file=near_cand_file, latitude=27, multiple_data_sets=new_color_at)

    far_file_list = [prefix + f'region_candidates/region_ra{int(round(ra*100))}_dec{int(round(dec*100))}_rad{int(round(region_rad*100))}_candidates.txt' for (ra, dec) in zip(ra_suc, dec_suc)]

    near_file_list = [prefix + f'region_candidates/region_ra{int(round(ra*100))}_dec{int(round(dec*100))}_rad{int(round(region_rad*100))}_candidates.txt' for (ra, dec) in zip(ra_near, dec_near)]

    print(len(far_file_list))
    print(new_color_at)

    new_all_sky(far_file_list, region_rad, near_plane_files=near_file_list, gal_plane_setting=gal_plane_setting, prefix=prefix, outfile=outfile, multiple_data_sets=new_color_at, labs=labs)


def extend_overdensity_bins(passing_indices_x, passing_indices_y, bin_range=5, maximum_range=1000):
    """Extend the list of spatial overdensities to the bins around overdense bins within range."""
    expanded_indices = [(x, y) for x in np.arange(-bin_range, bin_range + 1, dtype=int) for y in np.arange(-bin_range, bin_range + 1, dtype=int) if (x**2 + y**2 <= bin_range**2)]

    new_x, new_y = [], []
    for x, y in zip(passing_indices_x, passing_indices_y):
        for expanded_x, expanded_y in expanded_indices:
            if (expanded_x + x < maximum_range) & (expanded_y + y < maximum_range) & (expanded_x + x >= 0) & (expanded_y + y >= 0):
                new_x.append(expanded_x + x)
                new_y.append(expanded_y + y)

    passing_indices_x = np.concatenate((passing_indices_x, np.array(new_x, dtype=int)))
    passing_indices_y = np.concatenate((passing_indices_y, np.array(new_y, dtype=int)))

    return passing_indices_x, passing_indices_y


def get_gaia_ids(gaia_table, passing_spatial_x, passing_spatial_y, passing_pm_x, passing_pm_y, bin_size_spatial, bin_size_pm, just_spatial_indices=False):
    """Find and write GAIA ids of candidate objects."""
    ids = gaia_table['source_id']
    gaia_x = gaia_table['x'] * 180 / np.pi
    gaia_y = gaia_table['y'] * 180 / np.pi
    pm_ra = gaia_table['pmra']
    pm_dec = gaia_table['pmdec']

    good_indices_spatial = np.zeros(len(ids), dtype=bool)
    good_indices_pm = np.zeros(len(ids), dtype=bool)

    for x, y in zip(passing_spatial_x, passing_spatial_y):
        good_indices_spatial = good_indices_spatial | ((gaia_x > x) & (gaia_x < x + bin_size_spatial) & (gaia_y > y) & (gaia_y < y + bin_size_spatial))

    for x, y in zip(passing_pm_x, passing_pm_y):
        good_indices_pm = good_indices_pm | ((pm_ra > x) & (pm_ra < x + bin_size_pm) & (pm_dec > y) & (pm_dec < y + bin_size_pm))

    if just_spatial_indices is True:
        return good_indices_spatial

    good_indices = good_indices_spatial & good_indices_pm

    return ids[good_indices]


def cone_search(*, region_ra, region_dec, region_radius, radii, pm_radii, name=None, minimum_count_spatial=3, sigma_threshhold_spatial=3, minimum_count_pm=3, sigma_threshhold_pm=3, FLAG_search_pm_space=True, FLAG_plot=True, FLAG_restrict_pm=False, candidate_file_prefix='./candidates/', data_table_prefix='./candidates/regions', intersection_minima=[], extend_range=0):
    """Search region of sky."""
    # Set file paths
    infile = data_table_prefix + f'region_ra{round(region_ra*100)}_dec{round(region_dec*100)}_rad{round(region_radius*100)}.vot'
    outfile = candidate_file_prefix + f'region_candidates/region_ra{round(region_ra*100)}_dec{round(region_dec*100)}_rad{round(region_radius*100)}_candidates.txt'

    # first try to find existing input file
    try:
        with ZipFile(infile + '.zip') as myzip:
            print(myzip.namelist())
            with myzip.open(infile.split("/")[-1]) as myfile:
                gaia_table = Table.read(myfile, format='votable')
        print(f"Table loaded from: {infile}")
        print(f"Number of objects: {len(gaia_table)}")
    except FileNotFoundError:
        # If it can't be found, query GAIA and download then filter the table.
        job = gaia_region_search(region_ra, region_dec, outfile=infile, radius=region_radius)
        gaia_table = job.get_results()
        print("Finished querying Gaia")
        gaia_table = gaia_table[outside_of_galactic_plane(gaia_table['ra'], gaia_table['dec'])]
        print("Finished filtering Gaia table")
        # x-y values are projected coordinates (i.e. not sky coordinates)
        gaia_table['x'], gaia_table['y'] = azimuthal_equidistant_coordinates(gaia_table, region_ra, region_dec)
        print("Finished calculating x-y values done")
        print(f"Table dumped to: {infile}.zip")
        print(f"Number of objects: {len(gaia_table)}")
        gaia_table.write(infile, overwrite='True', format='votable')
        with ZipFile(infile + '.zip', 'w') as myzip:
            myzip.write(infile, arcname=infile.split("/")[-1], compress_type=ZIP_DEFLATED)
        os.remove(infile)

    # Get the convolved data for all radii
    convolved_data, xedges, yedges, X, Y, histo, histo_mask = convolve_spatial_histo(gaia_table, region_radius, radii)
    
    # Get passing candidate coordinates in projected (non-sky) coordinates
    passing_indices_x, passing_indices_y = cuts.histogram_overdensity_test(convolved_data, histo.shape, region_ra, region_dec, outfile, histo_mask, num_sigma=sigma_threshhold_spatial, repetition=minimum_count_spatial)

    od_test_result = False
    if len(passing_indices_x) > 0:
        od_test_result = True

    # Perform search in pm space if desired
    pm_test_result = False
    passing_indices_pm_x = []
    passing_indices_pm_y = []

    # Only search for pm overdensities using spatial overdensity objects
    gaia_table_restricted = gaia_table[:]
    if FLAG_restrict_pm is True:
        passing_indices_x, passing_indices_y = extend_overdensity_bins(passing_indices_x, passing_indices_y, bin_range=extend_range, maximum_range=len(xedges))
        extended_spatial_indices = get_gaia_ids(gaia_table, xedges[passing_indices_x], yedges[passing_indices_y], passing_indices_pm_x, passing_indices_pm_y, just_spatial_indices=True, bin_size_spatial=min(radii), bin_size_pm=min(pm_radii))
        gaia_table_restricted = gaia_table[extended_spatial_indices]

    # Search pm space for overdensities
    if FLAG_search_pm_space:
        convolved_data_pm, x_edges_pm, y_edges_pm, X_pm, Y_pm, histog, histog_mask = convolve_pm_histo(gaia_table_restricted, region_radius, pm_radii)
        passing_indices_pm_x, passing_indices_pm_y = cuts.pm_overdensity_test(convolved_data_pm, histog.shape, region_ra, region_dec, outfile, histog_mask, num_sigma=sigma_threshhold_pm, repetition=minimum_count_pm)

        if len(passing_indices_pm_x) > 0:
            pm_test_result = True

    # coordinate of center of bins
    min_radius = min(radii)
    passing_x = xedges[passing_indices_x] + min_radius / 2
    passing_y = yedges[passing_indices_y] + min_radius / 2
    min_pm_radius = min(pm_radii)
    passing_pm_x = x_edges_pm[passing_indices_pm_x] + min_pm_radius / 2
    passing_pm_y = y_edges_pm[passing_indices_pm_y] + min_pm_radius / 2

    overdense_objects = 0
    if pm_test_result is True & od_test_result is True:
        successful_object_ids = get_gaia_ids(gaia_table, xedges[passing_indices_x], yedges[passing_indices_y], x_edges_pm[passing_indices_pm_x], y_edges_pm[passing_indices_pm_y], min(radii), min(pm_radii))
        np.savetxt(outfile.rstrip('.txt') + '_ids.txt', successful_object_ids, header=f'# ids of objects in overdense bins for {name}')

        overdense_objects = len(successful_object_ids)

        # Coordinate transform back to coordinates on the sky
        passing_ra, passing_dec = inverse_azimuthal_equidistant_coordinates(np.deg2rad(passing_x), np.deg2rad(passing_y), np.deg2rad(region_ra), np.deg2rad(region_dec))

        # Create output file
        with open(outfile, 'w') as fil:
            fil.write(f'# successful candidates for region at {name} and radius {region_radius}')

        # Write successful sky coordinates
        with open(outfile, 'a') as outfl:
            for ra, dec in zip(passing_ra, passing_dec):
                outfl.write(f"{ra} {dec}\n")

    # plot the convolved data
    if FLAG_plot is True:
        convolved_histograms(convolved_data, (X, Y, histo), passingxy=[passing_x, passing_y], name=name, region_radius=region_radius, candidate_file_prefix=candidate_file_prefix)
        convolved_histograms_1d(convolved_data, (X, Y, histo), name=name, mask=histo_mask, region_radius=region_radius, candidate_file_prefix=candidate_file_prefix)
        convolved_pm_histograms(convolved_data_pm, (X_pm, Y_pm, histog), passingxy=[passing_pm_x, passing_pm_y], name=name, region_radius=5, candidate_file_prefix=candidate_file_prefix)

    return od_test_result, pm_test_result, overdense_objects


def main(main_args, input_file):
    """Search for dsph candidates."""
    import time
    start_time = time.time()

    count_pass_spatial = 0
    count_pass_pm = 0
    count_pass_both = 0
    count_total = 0
    count_od_intersection = 0

    known_dwarf_names = np.loadtxt("./the_search/tuning/tuning_known_dwarfs.txt", delimiter=",", dtype=str)[1:, 0]

    dwarfs = np.loadtxt(input_file, delimiter=" ", dtype=np.float64, comments='#')
    print(dwarfs)

    passing_dwarfs = []

    try:
        os.mkdir(main_args['candidate_file_prefix'])
    except OSError:
        pass

    try:
        os.mkdir(main_args['data_table_prefix'])
    except OSError:
        pass

    try:
        os.mkdir(main_args['candidate_file_prefix'] + 'region_candidates/')
    except OSError:
        pass

    try:
        os.mkdir(main_args['candidate_file_prefix'] + 'histos/')
    except OSError:
        pass

    for i, (ra, dec) in enumerate(dwarfs[:]):
        ra = float(ra)
        dec = float(dec)
        name = f"({ra}, {dec})"
        if input_file == 'the_search/tuning/tuning_known_dwarfs_no_name.txt':
            name = known_dwarf_names[i]

        main_args["name"] = name
        main_args["region_ra"] = ra
        main_args["region_dec"] = dec

        sp_pass, pm_pass, overdense_objects = cone_search(**main_args)
        if overdense_objects > 0: 
            count_od_intersection += 1

        if sp_pass is True:
            count_pass_spatial += 1
            with open(main_args['candidate_file_prefix'] + "successful_candidates_spatial.txt", 'a') as outfile:
                outfile.write(f"{ra} {dec}\n")

        if pm_pass is True:
            count_pass_pm += 1
            with open(main_args['candidate_file_prefix'] + "successful_candidates_pm.txt", 'a') as outfile:
                outfile.write(f"{ra} {dec}\n")

        if pm_pass is True and sp_pass is True:
            count_pass_both += 1
            passing_dwarfs.append((name, overdense_objects))
            print(f"Success: both tests passed")
            with open(main_args['candidate_file_prefix'] + "successful_candidates.txt", 'a') as outfile:
                outfile.write(f"{ra} {dec}\n")

        for intersection_minimum in main_args['intersection_minima']:
            if overdense_objects >= intersection_minimum:
                with open(main_args['candidate_file_prefix'] + f"successful_candidates_with_overlap_gte{intersection_minimum}.txt", 'a') as outfile:
                    outfile.write(f"{ra} {dec}\n")

        count_total += 1
        print(f"finished with dwarf {name}\t\t ({i}/{len(dwarfs)}) \n\n\n")

    print("Search parameters:")
    print(f'spatial count: {main_args["minimum_count_spatial"]}; spatial sigma: {main_args["sigma_threshhold_spatial"]}; pm count: {main_args["minimum_count_pm"]}; pm sigma: {main_args["sigma_threshhold_pm"]}')

    print("Passing dwarfs:")
    print(passing_dwarfs)

    print("Dwarf pass rate is")
    print(f"spatial {count_pass_spatial}/{count_total} = {count_pass_spatial/count_total}")
    print(f"PM count {count_pass_pm}/{count_total} = {count_pass_pm/count_total}")
    print(f"Both count {count_pass_both}/{count_total} = {count_pass_both/count_total}")
    print(f"Pass rate with overlapping objs in pm/spat bins {count_od_intersection}/{count_total} = {count_od_intersection/count_total}")

    print(f'\n\nAll results saved in {main_args["candidate_file_prefix"]}')

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main_args = {
            "region_radius": 1.0,
            "radii": [0.316, 0.1, 0.0316, 0.01, 0.00316],
            # "pm_radii": [1.5, 1.0, 0.5, 0.15],
            "pm_radii": [1.0, 0.316, 0.1, 0.0316, 0.01],
            "minimum_count_spatial": 3,
            "sigma_threshhold_spatial": 3,
            "minimum_count_pm": 3,
            "sigma_threshhold_pm": 3,
            "extend_range": 3,
            "FLAG_search_pm_space": True,
            "FLAG_plot": True,
            "FLAG_restrict_pm": True,
            "intersection_minima": [1, 2, 5, 10, 50],
            # "data_table_prefix": '/home/runburg/nfs_fs02/runburg/candidates/regions/'
            "data_table_prefix": './candidates/regions/'
            }

    main_args["candidate_file_prefix"] = f"./candidates/testing_trial{str(main_args['minimum_count_spatial'])}{str(main_args['sigma_threshhold_spatial'])}{str(main_args['minimum_count_pm'])}{str(main_args['sigma_threshhold_pm'])}_rad{str(int(main_args['region_radius']*100))}_small_pm_range{str(main_args['extend_range'])}/"
    # main_args['candidate_file_prefix'] = './candidates/'

    main(main_args, sys.argv[1])

    gal_plane_setting = 18
    # filter_then_plot(['./candidates/successful_candidates_north.txt', './candidates/successful_candidates_south.txt'])
    outfile = f"{str(main_args['minimum_count_spatial'])}{str(main_args['sigma_threshhold_spatial'])}{str(main_args['minimum_count_pm'])}{str(main_args['sigma_threshhold_pm'])}_rad{str(int(main_args['region_radius']*100))}"

    # filter_then_plot(['successful_candidates_with_overlap_gte10.txt'], prefix=main_args['candidate_file_prefix'], gal_plane_setting=gal_plane_setting, radius=main_args['region_radius'], outfile=f'all_sky_plot_{outfile}_intersection_10')
    counts = [1, 2, 5, 10, 50]
    # filter_then_plot([f'successful_candidates_with_overlap_gte{count}.txt' for count in counts], prefix=main_args['candidate_file_prefix'], gal_plane_setting=gal_plane_setting, radius=main_args['region_radius'], outfile=f'all_sky_plot_{outfile}_intersection', labs=counts)
