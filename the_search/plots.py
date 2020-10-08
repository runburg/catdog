#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make plots for dsph search project.

Author: Jack Runburg
Date: 11-09-2019 11:16


"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import glob
import matplotlib.patches as mpatches
try:
    from utils import inverse_azimuthal_equidistant_coordinates
except ModuleNotFoundError:
    from .utils import inverse_azimuthal_equidistant_coordinates
# from the_search.utils import inverse_azimuthal_equidistant_coordinates


def colorbar_for_subplot(fig, axs, cmap, image):
    """Place a colorbar by each plot.

    Helper function to place a color bar with nice spacing by the plot.


    Inputs:
        - fig: figure with the relevant axes
        - axs: axs to place the colorbar next to
        - cmap: color map for the colorbar
        - image: image for the colorbar

    Returns:
        - the colorbar object
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(axs)
    # Create the axis for the colorbar 
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # Create the colorbar
    cbar = fig.colorbar(image, cax=cax)
    # cm.ScalarMappable(norm=None, cmap=cmap),

    return cbar


def trim_axes(axes, N):
    """Trim the axes list to proper length.

    Helper function if too many subplots are present.


    Input:
        - axes: list of axes of the subplots
        - N: the number of subplots to keep

    Return:
        - list of retained axes
    """
    if N > 1:
        axes = axes.flat
        for ax in axes[N:]:
            ax.remove()
        return axes[:N]

    return [axes]


def plot_setup(rows, cols, d=0, buffer=(0.4, 0.4)):
    """Set mpl parameters for beautification.

    Make matplotlib pretty again!


    Input:
        - rows: number of rows of subplots
        - cols: number of columns of subplots
        - d: number of total subplots needed
        - buffer: tuple of white space around each subplot

    Returns:
        - figure object, list of subplot axes
    """
    # setup plot
    plt.close('all')

    # Format label sizes
    mpl.rcParams['axes.labelsize'] = 'medium'
    mpl.rcParams['ytick.labelsize'] = 'xx-small'
    mpl.rcParams['xtick.labelsize'] = 'xx-small'

    # Set white spaces
    mpl.rcParams['figure.subplot.wspace'] = buffer[0]
    mpl.rcParams['figure.subplot.hspace'] = buffer[1]

    # Choose font
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create figure
    figsize = (4 * cols + buffer[0], 3.5 * rows + buffer[1])
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    # Trim unnecessary axes
    if d != 0:
        axs = trim_axes(axs, d)

    return fig, axs


def convolved_histograms(convolved_data, histo_data, passingxy=None, name='dwarf', region_radius=0, candidate_file_prefix=''):
    """Make 2d histograms of convolved data.

    Create histograms of the objects in a given region as they are convolved.
    Meant to visualize the result of successful regions.


    Inputs:
        - convolved_data: list of tuples (radius, convolved_array) of the radius of the convolution kernel and the resulting array
        - histo_data: tuple (X, Y, histo) from unconvolved histogram
        - passingxy: list [passingx, passingy] of coordinates that pass the given test
        - name: name of dwarf/region
        - region_radius: radius in degrees of the region
    """
    from matplotlib import cm, colors

    # Unpack histogram data
    X, Y, histo = histo_data

    # Setup plot structure
    cols = 2
    d = len(convolved_data) + 1
    rows = d//2 + d % 2

    fig, axs = plot_setup(rows, cols, d)
    # fig.tight_layout()

    # Set bounds for color map
    vmin = 0
    vmax = np.nanmax(convolved_data[0][1])
    for _, cd in convolved_data:
        if np.nanmax(cd) > vmax:
            vmax = np.nanmax(cd)
    cmap = cm.magma
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)

    # Loop through the convolved data and plot
    for ax, (radius, convolved_array) in zip(axs, convolved_data):
        vmin = 0
        vmax = np.nanmax(convolved_array)
        normalize = colors.Normalize(vmin=vmin, vmax=vmax)
        ax.pcolormesh(X, Y, convolved_array.T, norm=normalize, cmap=cmap)

        ax.set_xlim(left=min(X[-1]), right=max(X[0]))
        ax.set_ylim(top=max(Y[-1]), bottom=min(Y[0]))

        # Label plot
        ax.set_title(f"2D tophat, radius={2*radius}")
        ax.set_xlabel("Relative ra [deg]")
        ax.set_ylabel("Relative dec [deg]")

        cbar = colorbar_for_subplot(fig, ax, cmap, cm.ScalarMappable(norm=normalize, cmap=cmap))
        # Overlay passing coordinates if any
        if passingxy is not None and ax == axs[-2]:
            ax.scatter(passingxy[0], passingxy[1], s=1, color='xkcd:bright teal')

    # Make the last plot the unconvolved histogram
    axs.flatten()[-1].pcolormesh(X, Y, histo.T, norm=normalize, cmap=cmap)
    axs.flatten()[-1].set_title("2D histogram, not convolved")

    # Add colorbars
    fig.suptitle(f"Convolved histogram for {name}")
    # fig.colorbar(cm.ScalarMappable(norm=normalize, cmap=cmap), ax=axs.ravel().tolist())

    # Save plot
    outfile = candidate_file_prefix + f'histos/{name}_histo_spatial.png'
    fig.savefig(outfile)

    print("saved to", outfile)


def convolved_histograms_1d(convolved_data, histo_data, name='dwarf', mask=None, region_radius=0, candidate_file_prefix=''):
    """Make 1d histogram of convolved data.

    Make plots of the 1d histogram of convolved data to visualize the overdensity.


    Inputs:
        - convolved_data: list of tuples (radius, convolved_array) of the radius of the convolution kernel and the resulting array
        - histo_data: tuple (X, Y, histo) from unconvolved histogram
        - name: name of dwarf/region
        - mask: mask array of bins to ignore in finding overdensitites
        - region_radius: radius in degrees of the region
    """
    from matplotlib import cm, colors

    # Unpack histogram data
    X, Y, histo = histo_data

    # Setup plot
    cols = 2
    d = len(convolved_data) + 1
    rows = d//2 + d % 2

    fig, axs = plot_setup(rows, cols, d)
    # fig.tight_layout()

    # Set bounds for colormap
    vmin = 0
    vmax = np.amax(histo)/10
    cmap = cm.magma
    normalize = colors.Normalize(vmin=vmin, vmax=vmax)

    # Loop through data and plot
    for ax, (radius, convolved_array) in zip(axs, convolved_data):
        hist_data, bins, _ = ax.hist(convolved_array.flatten()[(~np.isnan(convolved_array)).flatten()], density=False, bins=101, range=(0, np.nanmax(convolved_array)))

        # Label plot
        ax.set_yscale('log')
        ax.set_title(f"Bin counts,  conv. width={2*radius}")
        ax.set_ylabel("Frequency of counts")
        ax.set_xlabel("2d bin counts")

    # Plot the unconvolved histogram
    axs.flatten()[-1].pcolormesh(X, Y, histo.T, norm=normalize, cmap=cmap)
    axs.flatten()[-1].set_title("1D histogram, not convolved")

    fig.suptitle(f"1D Histogram for {name}")

    # Save figure
    fig.savefig(candidate_file_prefix + f'histos/{name}_histo_1d.png')


def new_all_sky(success_files, region_radius, near_plane_files=[], prefix='./candidates/', gal_plane_setting=15, outfile='all_sky_plot', multiple_data_sets=[], labs=[], color_max=100):
    """Plot candidates without Milky Way background."""
    print('plotting from success_files', len(success_files))
    ##############################
    # SET UP
    ##############################
    import astropy.coordinates as coord
    from astropy import units as u
    # from matplotlib import cm

    FLAG_plot_circles = False
    # FLAG_plot_region = False
    # FLAG_regions_from_file = True
    file_type = "png"

    region_rad_str = str(round(region_radius*100))
    # set up plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection="hammer")
    ax.set_facecolor("xkcd:black")
    ax.grid(color=lighten_color('xkcd:greyish blue', amount=1.5), zorder=0)
    ax.tick_params(axis='x', colors='xkcd:white')

    ##############################
    # PLOT BACKGROUND REGIONS
    ##############################
    # plot galactic plane
    ra_gal = np.linspace(0, 360, num=500)
    dec_gal = np.ones(500)*gal_plane_setting
    ra_icrs, dec_icrs = galactic_to_icrs(ra_gal, dec_gal)
    ra_icrs = coord.Angle(ra_icrs, unit='deg').wrap_at(180*u.degree)
    dec_icrs = coord.Angle(dec_icrs, unit='deg')
    ax.scatter(ra_icrs.radian, dec_icrs.radian, color='xkcd:steel gray', s=1, zorder=-1000)
    ra_icrs, dec_icrs = galactic_to_icrs(ra_gal, -dec_gal)
    ra_icrs = coord.Angle(ra_icrs, unit='deg').wrap_at(180*u.degree)
    dec_icrs = coord.Angle(dec_icrs, unit='deg')
    ax.scatter(ra_icrs.radian, dec_icrs.radian, color='xkcd:steel gray', s=1, zorder=-1000)

    # plot LMC
    ra_lmc, dec_lmc = get_points_of_circle(80.89, -69.76, 5)
    ra_lmc = coord.Angle(ra_lmc, unit='deg').wrap_at(180*u.degree)
    dec_lmc = coord.Angle(dec_lmc, unit='deg')
    ax.scatter(ra_lmc.radian, dec_lmc.radian, color='xkcd:steel gray', s=1, zorder=-1000)

    # plot SMC
    ra_smc, dec_smc = get_points_of_circle(13.16, -72.8, 2)
    ra_smc = coord.Angle(ra_smc, unit='deg').wrap_at(180*u.degree)
    dec_smc = coord.Angle(dec_smc, unit='deg')
    ax.scatter(ra_smc.radian, dec_smc.radian, color='xkcd:steel gray', s=1, zorder=-1000)
    # colors = ['xkcd:mauve', 'xkcd:coral', 'xkcd:pinkish purple', 'xkcd:tangerine', 'xkcd:vermillion', 'xkcd:tomato', 'xkcd:salmon', 'xkcd:dark peach', 'xkcd:marigold']

    ##############################
    # PLOT KNOWN DWARFS
    ##############################
    # Look for list of known dwarfs
    try:
        known = np.loadtxt('./the_search/tuning/tuning_known_dwarfs.txt', delimiter=',', dtype=str)
    except OSError:
        known = np.loadtxt('./dsph_search/the_search/tuning/tuning_known_dwarfs.txt', delimiter=',', dtype=str)

    # Get properties from files
    labels = [f"$\mathrm{{{know}}}$" for know in known[:, 0]]
    ra_known = known[:, 1].astype(np.float)
    dec_known = known[:, 2].astype(np.float)

    # Mutate data for plotting in ICRS coordinates
    ra = coord.Angle(ra_known*u.degree)
    ra = ra.wrap_at(180*u.degree)
    dec = coord.Angle(dec_known*u.degree)

    # Plot dwarf labels on the sky
    for (l, r, d) in zip(labels, ra, dec):
        ax.scatter(r.radian, d.radian, color='xkcd:light grey blue', marker=l, s=700, zorder=100)
    # print(len(glob.glob('./dsph_search/region_candidates/*.txt')))

    ##############################
    # PLOT ALL CANDIDATES
    ##############################
    # Plot the number of candidates
    ax.text(-150, -52, f"No. of cones: {len(success_files)}")

    # Get all the files with candidates
    file_list = success_files + near_plane_files

    # color the succesful cones by counts
    norm = mpl.colors.Normalize(vmin=0, vmax=np.log10(color_max))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), ax=ax, label="log10 of counts")

    # colors += ['xkcd:steel gray'] * len(near_plane_files)

    # Plot all the candidates in each file
    zorder = 500
    # print(len(file_list))
    i = 0
    for i, file in enumerate(file_list):
        if i < len(success_files):
            near = False
        else:
            near = True
    # for color, file in zip(colors, file_list):
        try:
            candidate_list = np.loadtxt(file, delimiter=" ")
            # print(candidate_list)
        except OSError:
            print('unable to load', file)
            i += 1
            continue
        # print(file)
        # print(len(candidate_list))

        # parse file name to get region coordinates
        file = file.split('/')[-1].split('_')
        # print(file)
        ra = float(file[1].strip('ra'))/100
        if ra < 0:
            ra += 180
        dec = float(file[2].strip('dec'))/100
        radius = float(file[3].strip('rad'))/100
        # print(ra, dec, radius)

        if FLAG_plot_circles is True:
            # get circle region around candidate
            circle_ra, circle_dec = get_points_of_circle(ra, dec, radius)

            # get center of circle region
            ra = coord.Angle(circle_ra*u.degree)
            ra = ra.wrap_at(180*u.degree)
            dec = coord.Angle(circle_dec*u.degree)

            # plot circular region
            ax.scatter(ra.radian, dec.radian, color=lighten_color(color, amount=1.6), s=0.1, zorder=50)

        # plot all the candidates in each file
        if len(candidate_list) >= 1:
            try:
                ra = coord.Angle(candidate_list[:, 0]*u.degree)
            except IndexError:
                # print(candidate_list)
                candidate_list = np.array([candidate_list])
            ra = coord.Angle(candidate_list[:, 0]*u.degree)
            ra = ra.wrap_at(180*u.degree)
            dec = coord.Angle(candidate_list[:, 1]*u.degree)
            if ~near:
                colors = list(cm.viridis(np.log10(candidate_list[:, 2])/np.log10(color_max)))
            else:
                colors = 'xkcd:steel gray'

            ax.scatter(ra.radian, dec.radian, color=colors, s=2, zorder=zorder)
        zorder += 1
    # save plot
    print('saved', prefix, outfile, file_type)
    fig.savefig(f"{prefix}{outfile}.{file_type}")


def get_points_of_circle(ra_center, dec_center, radius):
    """Get coordinates of circle for plotting."""
    n = 200
    coord_gen = np.linspace(0, 2*np.pi, num=n)

    radius = np.deg2rad(radius)
    x = radius * np.cos(coord_gen)
    y = radius * np.sin(coord_gen)

    return inverse_azimuthal_equidistant_coordinates(x, y, np.deg2rad(ra_center), np.deg2rad(dec_center))


def icrs_to_galactic(ra_icrs, dec_icrs):
    """Return galactic coordinates."""
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(ra_icrs, dec_icrs, unit='deg', frame='icrs')
    return np.array([coords.galactic.b.value, coords.galactic.l.value]).T


def galactic_to_icrs(ra_gal, dec_gal):
    """Return galactic coordinates."""
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(ra_gal, dec_gal, unit='deg', frame='galactic')
    return np.array([coords.icrs.ra, coords.icrs.dec])


def spherical_to_cartesian(ra, dec):
    """Get cartesian values from spherical."""
    ra_rad = ra * np.pi/180
    dec_rad = np.pi/2 - dec * np.pi/180
    return np.array([np.sin(dec_rad)*np.cos(ra_rad), np.sin(dec_rad)*np.sin(ra_rad), np.cos(dec_rad)])


def cartesian_to_spherical(vec):
    """Get spherical values from cartesian."""
    ra_rad = np.arctan2(vec[1], vec[0])
    dec_rad = np.arccos(vec[2]/np.linalg.norm(vec))
    if ra_rad < 0:
        ra_rad += 2*np.pi
    return np.array([ra_rad * 180/np.pi, 90 - dec_rad * 180/np.pi]).T


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.

    Input can be matplotlib color string, hex string, or RGB tuple. Make amount > 1 to darken.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if __name__ == '__main__':
    new_all_sky(3.16)
    # all_sky()
    # get_points_of_circle(30, 60, 5)
