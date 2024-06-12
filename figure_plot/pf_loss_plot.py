"""
    plot the topography (elevation & slope) of the primary forest distribution and lost primary forest
"""

import time
import numpy as np
import os
from os.path import join
import sys
import glob
import pandas as pd
from datetime import datetime
from osgeo import gdal, gdal_array, gdalconst
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import earthpy.plot as ep
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.interpolate import interpn
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from landcover_analysis.sf_change_dem import land_cover_map_read_hispaniola
from Basic_tools.Figure_plot import FP


def scatter_density_plot(x, y, x_label='', y_label='', title='', cmap=matplotlib.colormaps.get_cmap('Spectral_r'), figsize=(8, 6.5)):
    """
        plot the scatter density

        ref: answer by Guillaume in https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density/53865762
    """

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    tick_label_size = 16
    axis_label_size = 18
    title_label_size = 20
    tick_length = 4

    data, x_e, y_e = np.histogram2d(x, y, bins=[100, 100])
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    img = ax.scatter(x, y, c=z, s=10, cmap=cmap)

    ax.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True)
    ax.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True)

    ax.set_xlabel(x_label, size=axis_label_size)
    ax.set_ylabel(y_label, size=axis_label_size)

    ax.set_title(title, size=title_label_size)

    fig.colorbar(img, cmap=cmap)
    plt.show()

def get_pf_distribution_topography(land_cover, land_cover_flag, img_country, img_dem, img_slope):
    mask_haiti_pf_wet = (land_cover == land_cover_flag) & (img_country == 1)
    mask_dr_pf_wet = (land_cover == land_cover_flag) & (img_country == 2)

    haiti_pf_dem = img_dem[mask_haiti_pf_wet]
    haiti_pf_slope = img_slope[mask_haiti_pf_wet]

    dr_pf_dem = img_dem[mask_dr_pf_wet]
    dr_pf_slope = img_slope[mask_dr_pf_wet]

    return haiti_pf_dem, haiti_pf_slope, dr_pf_dem, dr_pf_slope


def get_pf_loss_topography(img_lc_1996, img_lc_2022, img_country_mask, img_dem, img_slope):

    mask_haiti_pf_wet_loss = (img_lc_1996 == 3) & (img_lc_2022 != 3) & (img_country_mask == 1)

    haiti_pwf_loss_elevation = img_dem[mask_haiti_pf_wet_loss]
    haiti_pwf_loss_slope = img_slope[mask_haiti_pf_wet_loss]

    mask_haiti_pf_dry_loss = (img_lc_1996 == 4) & (img_lc_2022 != 4) & (img_country_mask == 1)

    haiti_pdf_loss_elevation = img_dem[mask_haiti_pf_dry_loss]
    haiti_pdf_loss_slope = img_slope[mask_haiti_pf_dry_loss]

    mask_dr_pf_wet_loss = (img_lc_1996 == 3) & (img_lc_2022 != 3) & (img_country_mask == 2)

    dr_pwf_loss_elevation = img_dem[mask_dr_pf_wet_loss]
    dr_pwf_loss_slope = img_slope[mask_dr_pf_wet_loss]

    mask_dr_pf_dry_loss = (img_lc_1996 == 4) & (img_lc_2022 != 4) & (img_country_mask == 2)

    dr_pdf_loss_elevation = img_dem[mask_dr_pf_dry_loss]
    dr_pdf_loss_slope = img_slope[mask_dr_pf_dry_loss]

    return (haiti_pwf_loss_elevation, haiti_pwf_loss_slope, haiti_pdf_loss_elevation, haiti_pdf_loss_slope,
            dr_pwf_loss_elevation, dr_pwf_loss_slope, dr_pdf_loss_elevation, dr_pdf_loss_slope)

def get_density_info(x, y):
    data, x_e, y_e = np.histogram2d(x, y, bins=[100, 100])
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    return x, y, z

def plot_distribution_density(haiti_pwf_dem, haiti_pwf_slope, haiti_pdf_dem, haiti_pdf_slope,
                              x_label='', y_label='', title='',
                              cbar_1_label='primary wet forest',
                              cbar_2_label='primary dry forest',
                              output_flag=0,
                              output_folder='',
                              output_filename='',
                              figsize=(16, 10)
                              ):
    """
        plot the scatter density of the topography distribution of primary forest or lost PF in Haiti or DR
    """

    matplotlib.rcParams['font.family'] = "Arial"

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    tick_label_size = 32
    axis_label_size = 36
    cbar_tick_label_size = 20
    title_label_size = 30
    tick_length = 4

    # x, y, z = get_density_info(np.concatenate([haiti_pwf_dem, haiti_pdf_dem]), np.concatenate([haiti_pwf_slope, haiti_pdf_slope]))
    # img = ax.scatter(x, y, c=z, s=10, cmap=plt.cm.Greens)
    # cb = plt.colorbar(img, cmap=plt.cm.Greens)
    # cb.ax.tick_params(labelsize=tick_label_size)
    # cb.ax.set_ylabel('primary wet forest', size=axis_label_size)

    cmap = ListedColormap(matplotlib.colormaps.get_cmap('Greens')(np.linspace(0.2, 1, 100)))
    x, y, z = get_density_info(haiti_pwf_dem, haiti_pwf_slope)
    img = ax.scatter(x, y, c=z, s=10, cmap=cmap)
    cb = plt.colorbar(img, cmap=cmap)
    cb.ax.tick_params(labelsize=cbar_tick_label_size)
    cb.ax.set_ylabel(cbar_1_label, size=axis_label_size)

    cmap = ListedColormap(matplotlib.colormaps.get_cmap('Oranges')(np.linspace(0.2, 1, 100)))
    x, y, z = get_density_info(haiti_pdf_dem, haiti_pdf_slope)
    img = ax.scatter(x, y, c=z, s=10, cmap=cmap)
    cb = plt.colorbar(img, cmap=cmap)
    cb.ax.tick_params(labelsize=cbar_tick_label_size)
    cb.ax.set_ylabel(cbar_2_label, size=axis_label_size)

    ax.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True)
    ax.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True)

    ax.set_xlabel(x_label, size=axis_label_size)
    ax.set_ylabel(y_label, size=axis_label_size)

    ax.set_title(title, size=title_label_size)

    if output_flag == 0:
        plt.show()
    else:
        output = join(output_folder, '{}.jpg'.format(output_filename))
        plt.savefig(output, dpi=300)
        plt.close()


# def main():
if __name__ == '__main__':

    landcover_version = 'degrade_v2_refine_3_3'
    list_year = np.arange(1996, 2023)

    img_country_mask = gdal_array.LoadFile(join(rootpath, 'data', 'shapefile', 'landmask', 'countryid_hispaniola.tif'))

    img_dem = gdal_array.LoadFile(join(rootpath, 'data', 'dem', 'hispaniola_dem_info', 'dem_mosaic.tif'))
    img_dem[img_country_mask == 0] = np.nan
    img_slope = gdal_array.LoadFile(join(rootpath, 'data', 'dem', 'hispaniola_dem_info', 'slope_mosaic.tif'))
    img_slope[img_country_mask == 0] = np.nan

    ##

    img_lc_haiti_1996 = land_cover_map_read_hispaniola(1996, landcover_version, country_flag='hispaniola')
    img_lc_haiti_2022 = land_cover_map_read_hispaniola(2022, landcover_version, country_flag='hispaniola')

    haiti_pwf_dem, haiti_pwf_slope, dr_pwf_dem, dr_pwf_slope = get_pf_distribution_topography(img_lc_haiti_1996, 3, img_country_mask, img_dem, img_slope)
    haiti_pdf_dem, haiti_pdf_slope, dr_pdf_dem, dr_pdf_slope = get_pf_distribution_topography(img_lc_haiti_1996, 4, img_country_mask, img_dem, img_slope)

    (haiti_pwf_loss_dem, haiti_pwf_loss_slope, haiti_pdf_loss_dem, haiti_pdf_loss_slope,
     dr_pwf_loss_dem, dr_pwf_loss_slope, dr_pdf_loss_dem, dr_pdf_loss_slope) = get_pf_loss_topography(img_lc_haiti_1996, img_lc_haiti_2022, img_country_mask, img_dem, img_slope)

    # scatter_density_plot(haiti_pwf_dem, haiti_pwf_slope,
    #                      x_label='elevation', y_label='slope',
    #                      title='topography distribution of PWF \n in Haiti, 1996',
    #                      cmap=plt.cm.Greens)
    #
    # scatter_density_plot(haiti_pwf_loss_elevation, haiti_pwf_loss_slope,
    #                      x_label='elevation', y_label='slope',
    #                      title='topography distribution of PWF loss \n in Haiti from 1996 to 2022',
    #                      cmap=plt.cm.Oranges)

    # mask_haiti_pf_wet_loss = (img_lc_haiti_1996 == 4) & (img_lc_haiti_2022 != 4)
    #
    # elevation_haiti_pf_wet_loss = img_dem[mask_haiti_pf_wet_loss]
    # slope_haiti_pf_wet_loss = img_slope[mask_haiti_pf_wet_loss]
    #
    # scatter_density_plot(elevation_haiti_pf_wet_loss, slope_haiti_pf_wet_loss,
    #                      x_label='elevation', y_label='slope',
    #                      title='topography distribution of PDF loss \n in Haiti from 1996 to 2022',
    #                      cmap=plt.cm.Greens)

    ##

    plot_distribution_density(haiti_pwf_dem, haiti_pwf_slope, haiti_pdf_dem, haiti_pdf_slope,
                              x_label='elevation', y_label='slope',
                              title='topography distribution of PF in Haiti, 1996')

    plot_distribution_density(dr_pwf_dem, dr_pwf_slope, dr_pdf_dem, dr_pdf_slope,
                              x_label='elevation', y_label='slope',
                              title='topography distribution of PF in DR, 1996')

    ##

    plot_distribution_density(haiti_pwf_dem, haiti_pwf_slope, haiti_pdf_dem, haiti_pdf_slope,
                              x_label='elevation', y_label='slope',
                              title='topography distribution of PF in Haiti, 1996',
                              output_flag=1,
                              output_folder=r'C:\Users\64937\OneDrive\LCM_biodiversity\manuscript\figure\pf_topography',
                              output_filename='v4_1996_haiti_pf_distribution')

    plot_distribution_density(dr_pwf_dem, dr_pwf_slope, dr_pdf_dem, dr_pdf_slope,
                              x_label='elevation', y_label='slope',
                              title='topography distribution of PF in DR, 1996',
                              output_flag=1,
                              output_folder=r'C:\Users\64937\OneDrive\LCM_biodiversity\manuscript\figure\pf_topography',
                              output_filename='v4_1996_dr_pf_distribution')

    ##

    # plot_distribution_density(haiti_pwf_loss_dem, haiti_pwf_loss_slope, haiti_pdf_loss_dem, haiti_pdf_loss_slope,
    #                           x_label='elevation', y_label='slope',
    #                           title='topography distribution of lost PF from 1996 to 2022 in Haiti',
    #                           output_flag=1,
    #                           output_folder=r'C:\Users\64937\OneDrive\LCM_biodiversity\manuscript\figure\pf_topography',
    #                           output_filename='v4_haiti_pf_loss'
    #                           )
    #
    # plot_distribution_density(dr_pwf_loss_dem, dr_pwf_loss_slope, dr_pdf_loss_dem, dr_pdf_loss_slope,
    #                           x_label='elevation', y_label='slope',
    #                           title='topography distribution of lost PF from 1996 to 2022 in DR',
    #                           output_flag=1,
    #                           output_folder=r'C:\Users\64937\OneDrive\LCM_biodiversity\manuscript\figure\pf_topography',
    #                           output_filename='v4_dr_pf_loss'
    #                           )

    ##
    # x_label = 'elevation'
    # y_label = 'slope'
    # title = 'topography distribution of PF in Haiti, 1996'
    #
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14, 10))
    # tick_label_size = 16
    # axis_label_size = 20
    # title_label_size = 22
    # tick_length = 4
    #
    # # x, y, z = get_density_info(np.concatenate([haiti_pwf_dem, haiti_pdf_dem]), np.concatenate([haiti_pwf_slope, haiti_pdf_slope]))
    # # img = ax.scatter(x, y, c=z, s=10, cmap=plt.cm.Greens)
    # # cb = plt.colorbar(img, cmap=plt.cm.Greens)
    # # cb.ax.tick_params(labelsize=tick_label_size)
    # # cb.ax.set_ylabel('primary wet forest', size=axis_label_size)
    #
    # from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    #
    # cmap = ListedColormap(matplotlib.colormaps.get_cmap('Greens')(np.linspace(0.3, 1, 100)))
    # x, y, z = get_density_info(haiti_pwf_dem, haiti_pwf_slope)
    # img = ax.scatter(x, y, c=z, s=10, cmap=cmap)
    # cb = plt.colorbar(img, cmap=cmap)
    # cb.ax.tick_params(labelsize=tick_label_size)
    # cb.ax.set_ylabel('primary wet forest', size=axis_label_size)
    #
    # cmap = ListedColormap(matplotlib.colormaps.get_cmap('Oranges')(np.linspace(0.3, 1, 100)))
    # x, y, z = get_density_info(haiti_pdf_dem, haiti_pdf_slope)
    # img = ax.scatter(x, y, c=z, s=10, cmap=cmap)
    # cb = plt.colorbar(img, cmap=cmap)
    # cb.ax.tick_params(labelsize=tick_label_size)
    # cb.ax.set_ylabel('primary dry forest', size=axis_label_size)
    #
    # ax.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True)
    # ax.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True)
    #
    # ax.set_xlabel(x_label, size=axis_label_size)
    # ax.set_ylabel(y_label, size=axis_label_size)
    #
    # ax.set_title(title, size=title_label_size)
    #
    # plt.show()
