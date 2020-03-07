#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script for running the dSph search and controlling output.

Author: Jack Runburg
Date: 22-08-2019 14:03

"""

import glob
import warnings
import numpy as np
import sys, os
from zipfile import ZipFile, ZIP_DEFLATED
from astropy.table import Table
from the_search.dwarf import Dwarf
from the_search import cuts
from the_search.utils import fibonnaci_sphere, get_cone_in_region, gaia_region_search, outside_of_galactic_plane, azimuthal_equidistant_coordinates, inverse_azimuthal_equidistant_coordinates, get_window_function, convolve_spatial_histo, convolve_pm_histo
from the_search.plots import get_points_of_circle, convolved_histograms, convolved_histograms_1d, new_all_sky

warnings.filterwarnings("ignore")


def look_at_tuned_parameter_values(plot=False):
    """Check params from tuning."""
    dwarflist = []
    rando = []
    for dwa in np.loadtxt('the_search/tuning/tuning_known_dwarfs_old_names.txt', dtype=str, delimiter=','):
        dwarflist.append([dwa[1].astype(np.float), dwa[2].astype(np.float), dwa[0]])
    for ran in np.loadtxt('./dsph_search/the_search/tuning/tuning_random.txt', delimiter=','):
        rando.append([ran[0].astype(np.float), ran[1].astype(np.float)])

    # Execute cuts.
    cutlist = [100]
    set_of_dwarfs = [dwarflist, rando]
    for list_of_dwarfs, known, label in zip(set_of_dwarfs, [True, False], ['Known', 'Random']):
        dwarfpass = 0
        for dwarf in load_sample_dwarfs(list_of_dwarfs, known=known, path='./dsph_search/the_search/tuning/test_candidates'):
            for cut in cutlist:
                pass1 = cuts.proper_motion_test(dwarf, cut=cut, print_to_stdout=True, **params)
                pass2 = cuts.angular_density_test(dwarf, print_to_stdout=True, **params)
                dwarfpass += pass1 & pass2

            if plot is True:
                dwarf.accepted(plot)

        print(f'{label} dwarf pass rate \t{dwarfpass}/{len(list_of_dwarfs)}')

    # randompass = 0
    # for dwarf in load_sample_dwarfs(rando, known=, path='the_search/tuning/test_candidates'):
    #     for cut in cutlist:
    #         pass1 = cuts.proper_motion_test(dwarf, cut=cut, print_to_stdout=True, **params)
    #         pass2 = cuts.angular_density_test(dwarf, print_to_stdout=True, **params)
    #         randompass += pass1 & pass2
    #
    #     if plot is True:
    #         dwarf.accepted(plot)

    # print(f'Random pass rate \t{randompass}/{len(rando)}')
    print(params)


def write_candidate_coords():
    """Write out positions of dwarf candidates."""
    with open('candidate_coords.txt', 'w') as outfile:
        for file in glob.glob('./candidates/*'):
            ra, _, dec = file.rpartition('/')[-1].partition('_')
            outfile.write(str(round(float(ra)/100, 2)) + ' ' + str(round(float(dec)/100, 2)) + '\n')


def random_poisson(param_args):
    """Generate random poisson GAIA data for testing."""
    import matplotlib.pyplot as plt
    # # test_cases = [(radius1, radius2) for radius1 in radii for radius2 in radii+[region_radius] if radius1 < radius2]
    # # print(test_cases)

    region_ra, region_dec, region_radius, num_cones, *radii = [float(arf) for arf in param_args[1:]]

    radii = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    region_radius = 0.05
    num_pts = 20000

    sigma_threshhold = 5
    minimum_count = 3
    x = np.random.uniform(-region_radius, region_radius, num_pts)
    y = np.random.uniform(-region_radius, region_radius, num_pts)

    for radius in radii:
        poisson_sd = np.sqrt(num_pts * radius**2/region_radius**2)
        print(poisson_sd)
        histo, xedges, yedges = np.histogram2d(x, y, bins=region_radius//radius)
        bin_width = (xedges[1]-xedges[0])/2
        passing_indices_y, passing_indices_x = np.argwhere((histo > poisson_sd*sigma_threshhold) & (histo > minimum_count)).T
        passing_ra, passing_dec = xedges[passing_indices_x]+bin_width, yedges[passing_indices_y]+bin_width

        # with open(outfile, 'w') as outfl:
        #     for ra, dec in zip(passing_ra, passing_dec):
        #         outfl.write(f"{ra} {dec}")

    for radius in radii:
        fig, ax = plt.subplots()
        ax.hist2d(x, y, bins=region_radius//radius)
        fig.savefig(f'poisson_simulation/poisson_plot_{radius}.pdf')

    fig, ax = plt.subplots()
    ax.scatter(passing_ra, passing_dec, s=1)
    # ax.scatter(*get_points_of_circle(region_ra, region_dec, region_radius).T, s=1)
    fig.savefig('poisson_simulation/poisson_passing_coords_001.pdf')


def filter_then_plot(infiles):
    """Create all sky plot and filter candidates."""
    from the_search.utils import cut_out_candidates_close_to_plane_and_slmc
    from the_search.plots import new_all_sky

    region_rad = 3.16
    coord_list = np.concatenate([np.loadtxt(infile, delimiter=" ") for infile in infiles])

    filtered_cand_file = "successful_candidates_filtered.txt"
    near_cand_file = "successful_candidates_near.txt"
    ra_suc, dec_suc, ra_near, dec_near = cut_out_candidates_close_to_plane_and_slmc(coord_list[:, 0], coord_list[:, 1], far_file=filtered_cand_file, near_file=near_cand_file, latitude=27)

    far_file_list = [f'./region_candidates/region_ra{int(round(ra*100))}_dec{int(round(dec*100))}_rad{int(round(region_rad*100))}_candidates.txt' for (ra, dec) in zip(ra_suc, dec_suc)]

    near_file_list = [f'./region_candidates/region_ra{int(round(ra*100))}_dec{int(round(dec*100))}_rad{int(round(region_rad*100))}_candidates.txt' for (ra, dec) in zip(ra_near, dec_near)]

    new_all_sky(far_file_list, region_rad, near_plane_files=near_file_list)


def cone_search(*, region_ra, region_dec, region_radius, radii, pm_radii, name=None, minimum_count=3, sigma_threshhold=3, FLAG_search_pm_space=True, FLAG_plot=True):
    """Search region of sky."""
    # Give a default name based on position
    if name is None:
        name = f"({region_ra}, {region_dec})"

    # Set input/output paths
    infile = f'regions/region_ra{round(region_ra*100)}_dec{round(region_dec*100)}_rad{round(region_radius*100)}.vot'
    outfile = f'region_candidates/region_ra{round(region_ra*100)}_dec{round(region_dec*100)}_rad{round(region_radius*100)}_candidates.txt'

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
    passing_indices_x, passing_indices_y = cuts.histogram_overdensity_test(convolved_data, histo.shape, region_ra, region_dec, outfile, histo_mask, num_sigma=(sigma_threshhold-1), repetition=minimum_count)

    min_radius = min(radii)
    passing_x = xedges[passing_indices_x] + min_radius/2  # coordinate of center of bins
    passing_y = yedges[passing_indices_y] + min_radius/2

    od_test_result = False
    if len(passing_indices_x) > 0:
        od_test_result = True

    # Perform search in pm space if desired
    pm_test_result = True
    if FLAG_search_pm_space & od_test_result:
        convolved_data_pm, _, _, _, _, histog, histog_mask = convolve_pm_histo(gaia_table, region_radius, radii)
        pm_test_result = cuts.pm_overdensity_test(convolved_data_pm, histog.shape, region_ra, region_dec, outfile, histog_mask, num_sigma=sigma_threshhold, repetition=minimum_count)

    if pm_test_result is True:
        # Coordinate transform back to coordinates on the sky
        passing_ra, passing_dec = inverse_azimuthal_equidistant_coordinates(np.deg2rad(passing_x), np.deg2rad(passing_y), np.deg2rad(region_ra), np.deg2rad(region_dec))

        # plot the convolved data
        if FLAG_plot is True:
            convolved_histograms(convolved_data, (X, Y, histo), passingxy=[passing_x, passing_y], name=name, region_radius=region_radius)
            convolved_histograms_1d(convolved_data, (X, Y, histo), name=name, mask=histo_mask, region_radius=region_radius)

        # Create output file
        with open(outfile, 'w') as fil:
            fil.write(f'# successful candidates for region at ({region_ra}, {region_dec}) and radius {region_radius}')

        # Write successful sky coordinates
        with open(outfile, 'a') as outfl:
            for ra, dec in zip(passing_ra, passing_dec):
                outfl.write(f"{ra} {dec}\n")

    return od_test_result, pm_test_result


def main(param_args):
    """Search for dsph candidates."""
    import time
    start_time = time.time()

    _, input_file, region_radius, *radii = param_args
    radii = [float(radius) for radius in radii]
    region_radius = float(region_radius)

    count_pass_spatial = 0
    count_pass_pm = 0
    count_total = 0

    dwarfs = np.loadtxt(input_file, delimiter=" ", dtype=np.float64, comments='#')
    print(dwarfs)
    for i, (ra, dec) in enumerate(dwarfs[:]):
        ra = float(ra)
        dec = float(dec)
        name = f"({ra}, {dec})"

        main_args = {"region_ra": ra,
                     "region_dec": dec,
                     "region_radius": region_radius,
                     "radii": radii,
                     "pm_radii": radii,
                     "minimum_count": 3,
                     "sigma_threshhold": 3,
                     "name": name,
                     "FLAG_search_pm_space": True,
                     "FLAG_plot": False
                     }

        sp_pass, pm_pass = cone_search(**main_args)

        if sp_pass is True:
            count_pass_spatial += 1
        if pm_pass is True and sp_pass is True:
            count_pass_pm += 1
            print(f"Success: both tests passed")
            with open("successful_candidates.txt", 'a') as outfile:
                outfile.write(f"{ra} {dec}\n")
        count_total += 1
        print(f"finished with dwarf {name}\n\n\n")

    print("Dwarf pass rate is")
    print(f"Spatial {count_pass_spatial}/{count_total} = {count_pass_spatial/count_total}")
    print(f"PM count {count_pass_pm}/{count_total} = {count_pass_pm/count_total}")

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main(sys.argv)
    filter_then_plot(['successful_candidates_north.txt', 'successful_candidates_south.txt'])

 print(dra.gaia_data[-1][-1][[1,2,3]])
