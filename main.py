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
from the_search.plots import convolved_histograms, convolved_histograms_1d, new_all_sky

warnings.filterwarnings("ignore")


def filter_then_plot(infiles, prefix='./candidates/', gal_plane_setting=15):
    """Create all sky plot and filter candidates."""
    from the_search.utils import cut_out_candidates_close_to_plane_and_slmc

    region_rad = 3.16
    coord_list = np.concatenate([np.loadtxt(prefix + infile, delimiter=" ") for infile in infiles])

    filtered_cand_file = prefix + "successful_candidates_filtered.txt"
    near_cand_file = prefix + "successful_candidates_near.txt"

    ra_suc, dec_suc, ra_near, dec_near = cut_out_candidates_close_to_plane_and_slmc(coord_list[:, 0], coord_list[:, 1], far_file=filtered_cand_file, near_file=near_cand_file, latitude=27)

    far_file_list = [prefix + f'region_candidates/region_ra{int(round(ra*100))}_dec{int(round(dec*100))}_rad{int(round(region_rad*100))}_candidates.txt' for (ra, dec) in zip(ra_suc, dec_suc)]

    near_file_list = [prefix + f'region_candidates/region_ra{int(round(ra*100))}_dec{int(round(dec*100))}_rad{int(round(region_rad*100))}_candidates.txt' for (ra, dec) in zip(ra_near, dec_near)]

    print(len(far_file_list))

    new_all_sky(far_file_list, region_rad, near_plane_files=near_file_list, gal_plane_setting=gal_plane_setting, prefix=prefix)


def get_gaia_ids():
    """Find and write GAIA ids of candidate objects."""
    pass


def cone_search(*, region_ra, region_dec, region_radius, radii, pm_radii, name=None, minimum_count_spatial=3, sigma_threshhold_spatial=3, minimum_count_pm=3, sigma_threshhold_pm=3, FLAG_search_pm_space=True, FLAG_plot=True, candidate_file_prefix='./candidates/', data_table_prefix='./candidates/regions'):
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

    min_radius = min(radii)
    passing_x = xedges[passing_indices_x] + min_radius / 2  # coordinate of center of bins
    passing_y = yedges[passing_indices_y] + min_radius / 2

    od_test_result = False
    if len(passing_indices_x) > 0:
        od_test_result = True

    # Perform search in pm space if desired
    pm_test_result = True
    if FLAG_search_pm_space:
        convolved_data_pm, _, _, _, _, histog, histog_mask = convolve_pm_histo(gaia_table, region_radius, radii)
        pm_test_result = cuts.pm_overdensity_test(convolved_data_pm, histog.shape, region_ra, region_dec, outfile, histog_mask, num_sigma=sigma_threshhold_pm, repetition=minimum_count_pm)

    if pm_test_result is True & od_test_result is True:
        # Coordinate transform back to coordinates on the sky
        passing_ra, passing_dec = inverse_azimuthal_equidistant_coordinates(np.deg2rad(passing_x), np.deg2rad(passing_y), np.deg2rad(region_ra), np.deg2rad(region_dec))

        # Create output file
        with open(outfile, 'w') as fil:
            fil.write(f'# successful candidates for region at ({region_ra}, {region_dec}) and radius {region_radius}')

        # Write successful sky coordinates
        with open(outfile, 'a') as outfl:
            for ra, dec in zip(passing_ra, passing_dec):
                outfl.write(f"{ra} {dec}\n")

    # plot the convolved data
    if FLAG_plot is True:
        convolved_histograms(convolved_data, (X, Y, histo), passingxy=[passing_x, passing_y], name=name, region_radius=region_radius)
        convolved_histograms_1d(convolved_data, (X, Y, histo), name=name, mask=histo_mask, region_radius=region_radius)

    return od_test_result, pm_test_result


def main(main_args, input_file):
    """Search for dsph candidates."""
    import time
    start_time = time.time()

    count_pass_spatial = 0
    count_pass_pm = 0
    count_pass_both = 0
    count_total = 0

    known_dwarf_names = np.loadtxt("./the_search/tuning/tuning_known_dwarfs.txt", delimiter=",", dtype=str)[:, 0]

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

    for i, (ra, dec) in enumerate(dwarfs[:]):
        ra = float(ra)
        dec = float(dec)
        name = f"({ra}, {dec})"
        if input_file == 'the_search/tuning/tuning_known_dwarfs_no_name.txt':
            name = known_dwarf_names[i]

        main_args["name"] = name
        main_args["region_ra"] = ra
        main_args["region_dec"] = dec

        sp_pass, pm_pass = cone_search(**main_args)

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
            passing_dwarfs.append(name)
            print(f"Success: both tests passed")
            with open(main_args['candidate_file_prefix'] + "successful_candidates.txt", 'a') as outfile:
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

    print(f'\n\nAll results saved in {main_args["candidate_file_prefix"]}')

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main_args = {
                    "region_radius": 1.0,
                    "radii": [0.316, 0.1, 0.0316, 0.01, 0.00316],
                    "pm_radii": [0.316, 0.1, 0.0316, 0.01, 0.00316],
                    "minimum_count_spatial": 3,
                    "sigma_threshhold_spatial": 3,
                    "minimum_count_pm": 3,
                    "sigma_threshhold_pm": 3,
                    "FLAG_search_pm_space": True,
                    "FLAG_plot": False,
                    "data_table_prefix": '/home/runburg/nfs_fs02/runburg/candidates/regions/'
                }

    main_args["candidate_file_prefix"] = f"./candidates/trial{str(main_args['minimum_count_spatial'])}{str(main_args['sigma_threshhold_spatial'])}{str(main_args['minimum_count_pm'])}{str(main_args['sigma_threshhold_pm'])}_rad{str(int(main_args['region_radius']*100))}/"

    main(main_args, sys.argv[1])

    gal_plane_setting = 18
    # filter_then_plot(['./candidates/successful_candidates_north.txt', './candidates/successful_candidates_south.txt'])
    # filter_then_plot(['successful_candidates.txt'], prefix=main_args['candidate_file_prefix', gal_plane_setting=gal_plane_setting)

