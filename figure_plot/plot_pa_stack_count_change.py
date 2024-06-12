"""
plot the primary forest change in the protected area
"""

import numpy as np
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as plticker
import seaborn as sns
import os
import sys

sns.set_theme()

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def scatter_plot(x, pf_inside_pa, pf_outside_pa, x_label='', y_label='', plot_name='test'):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(12, 7))

    axes_linewidth = 1.5
    matplotlib.rcParams['axes.linewidth'] = axes_linewidth
    for i in axes.spines.values():
        i.set_linewidth(axes_linewidth)

    matplotlib.rcParams['font.family'] = 'arial'
    labelsize = 22
    xaxis_label_size = 24
    title_label_size = 28
    legend_fontsize = 27
    ticklength = 6

    axes.scatter(x, pf_inside_pa, marker="o", c='#ff474c', label='PF inside protected area')
    axes.scatter(x, pf_outside_pa, marker="o", c='#0165fc', label='PF outside protected area')

    axes.tick_params('x', labelsize=labelsize, direction='out', length=ticklength, width=axes_linewidth, bottom=True, which='major')
    axes.tick_params('y', labelsize=labelsize, direction='out', length=ticklength, width=axes_linewidth, left=True, which='major')

    axes.set_xlabel(x_label, size=xaxis_label_size)
    axes.set_ylabel(y_label, size=xaxis_label_size)

    axes.set_xlim(1993, 2023)

    plt.ticklabel_format(style='plain')

    plt.legend(loc='best', fontsize=20)

    plt.title(plot_name, fontsize=title_label_size)
    plt.tight_layout()
    plt.show()


def scatter_subplot_plot(axes, x, pf_inside_pa, pf_outside_pa, x_label, y_label, plot_name):

    axes_linewidth = 1.5
    matplotlib.rcParams['axes.linewidth'] = axes_linewidth
    for i in axes.spines.values():
        i.set_linewidth(axes_linewidth)

    matplotlib.rcParams['font.family'] = 'arial'
    labelsize = 22
    xaxis_label_size = 24
    title_label_size = 28
    legend_fontsize = 27
    ticklength = 6

    axes.scatter(x, pf_inside_pa, marker="o", c='#ff474c', label='PF inside protected area')
    axes.scatter(x, pf_outside_pa, marker="o", c='#0165fc', label='PF outside protected area')

    axes.tick_params('x', labelsize=labelsize, direction='out', length=ticklength, width=axes_linewidth, bottom=True, which='major')
    axes.tick_params('y', labelsize=labelsize, direction='out', length=ticklength, width=axes_linewidth, left=True, which='major')

    axes.set_xlabel(x_label, size=xaxis_label_size)
    axes.set_ylabel(y_label, size=xaxis_label_size)

    axes.set_xlim(1993, 2023)

    plt.ticklabel_format(style='plain')

    plt.legend(loc='best', fontsize=20)

    plt.title(plot_name, fontsize=title_label_size)
    plt.tight_layout()
    plt.show()


def pf_pa_stacked_plot(axes,
                       list_year,
                       values_plot,
                       x_label='Year',
                       y_label='Primary forest area ($\mathregular{km^2}$)',
                       title=None,
                       output_flag=0,
                       output_folder=None,
                       output_filename=None,
                       legend_flag=False,
                       ):
    """
        plot the stacked figure to show the PF inside and outside the protected area change
    """

    list_color = ['#1d6533', '#6ca966', '#b3afa4', '#929591']

    # sns.set_style('white')
    sns.set_theme()
    # fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(20, 12))

    legend_label_size = 26
    tick_label_size = 26
    xaxis_label_size = 30
    title_label_size = 34
    ticklength = 6
    axes_linewidth = 1.5

    matplotlib.rcParams['axes.linewidth'] = axes_linewidth
    for i in axes.spines.values():
        i.set_linewidth(axes_linewidth)
    matplotlib.rcParams['font.family'] = 'arial'

    for i in range(0, 2):
        axes.bar(list_year, values_plot[i], bottom=np.sum(values_plot[:i], axis=0))

    axes.tick_params('x', labelsize=tick_label_size, direction='out', length=ticklength, width=axes_linewidth, bottom=True, which='major')
    axes.tick_params('y', labelsize=tick_label_size, direction='out', length=ticklength, width=axes_linewidth, left=True, which='major')

    axes.set_xlabel(x_label, size=xaxis_label_size)
    axes.set_ylabel(y_label, size=xaxis_label_size)

    axes.xaxis.set_major_locator(plticker.MultipleLocator(base=2.0))

    plt.title(title, fontsize=title_label_size)
    plt.tight_layout()

    labels_legend = ['PF inside protected area', 'PF outside protected area']
    # axes.legend(labels=labels_legend, bbox_to_anchor=(0.5, -0.3), ncol=4, loc='lower center', fontsize=legend_label_size)
    if legend_flag:
        axes.legend(labels=labels_legend, loc='upper right', fontsize=legend_label_size)

    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.tight_layout()
    plt.show()

    if output_flag == 0:
        plt.show()
    else:
        output_file = join(output_folder, '{}.jpg'.format(output_filename))
        plt.savefig(output_file, dpi=300)
        plt.close()


# def main():
if __name__ == '__main__':

    landcover_version = 'degrade_v2_refine_3_3'

    # output_path_matrix = r'K:\LCM_diversity\results\land_change_modelling\hispaniola\predict_matrix_dataframe'

    filename_percentile = join(rootpath, 'results', 'protect_area_analysis', landcover_version,
                               'pa_num_count_{}.xlsx'.format(landcover_version))
    sheet_pa = pd.read_excel(filename_percentile)
    # print(sheet_hispaniola)

    list_year = sheet_pa['year'].values
    ##
    haiti_pf_inside_pct = sheet_pa['haiti_pf_inside_pa'] / sheet_pa['haiti_pf_num']
    haiti_pf_outside_pct = sheet_pa['haiti_pf_outside_pa'] / sheet_pa['haiti_pf_num']

    dr_pf_inside_pct = sheet_pa['dr_pf_inside_pa'] / sheet_pa['dr_pf_num']
    dr_pf_outside_pct = sheet_pa['dr_pf_outside_pa'] / sheet_pa['dr_pf_num']

    print(np.nanmean(haiti_pf_inside_pct), np.nanmean(haiti_pf_outside_pct))
    print(np.nanmean(dr_pf_inside_pct), np.nanmean(dr_pf_outside_pct))

    ##
    haiti_pf_wet_loss_inside_pct = (sheet_pa['haiti_pf_wet_inside_pa'].values[0] - sheet_pa['haiti_pf_wet_inside_pa'].values[-1]) / sheet_pa['haiti_pf_wet_inside_pa'].values[0] * 100
    haiti_pf_wet_loss_outside_pct = (sheet_pa['haiti_pf_wet_outside_pa'].values[0] - sheet_pa['haiti_pf_wet_outside_pa'].values[-1]) / sheet_pa['haiti_pf_wet_outside_pa'].values[0] * 100

    haiti_pf_dry_loss_inside_pct = (sheet_pa['haiti_pf_dry_inside_pa'].values[0] - sheet_pa['haiti_pf_dry_inside_pa'].values[-1]) / sheet_pa['haiti_pf_dry_inside_pa'].values[0] * 100
    haiti_pf_dry_loss_outside_pct = (sheet_pa['haiti_pf_dry_outside_pa'].values[0] - sheet_pa['haiti_pf_dry_outside_pa'].values[-1]) / sheet_pa['haiti_pf_dry_outside_pa'].values[0] * 100

    dr_pf_wet_loss_inside_pct = (sheet_pa['dr_pf_wet_inside_pa'].values[0] - sheet_pa['dr_pf_wet_inside_pa'].values[-1]) / sheet_pa['dr_pf_wet_inside_pa'].values[0] * 100
    dr_pf_wet_loss_outside_pct = (sheet_pa['dr_pf_wet_outside_pa'].values[0] - sheet_pa['dr_pf_wet_outside_pa'].values[-1]) / sheet_pa['dr_pf_wet_outside_pa'].values[0] * 100

    dr_pf_dry_loss_inside_pct = (sheet_pa['dr_pf_dry_inside_pa'].values[0] - sheet_pa['dr_pf_dry_inside_pa'].values[-1]) / sheet_pa['dr_pf_dry_inside_pa'].values[0] * 100
    dr_pf_dry_loss_outside_pct = (sheet_pa['dr_pf_dry_outside_pa'].values[0] - sheet_pa['dr_pf_dry_outside_pa'].values[-1]) / sheet_pa['dr_pf_dry_outside_pa'].values[0] * 100

    print(haiti_pf_wet_loss_inside_pct)
    print(haiti_pf_wet_loss_outside_pct)
    print(haiti_pf_dry_loss_inside_pct)
    print(haiti_pf_dry_loss_outside_pct)
    print(dr_pf_wet_loss_inside_pct)
    print(dr_pf_wet_loss_outside_pct)
    print(dr_pf_dry_loss_inside_pct)
    print(dr_pf_dry_loss_outside_pct)

    ##

    # x_label = 'Year'
    # y_label = 'Primary forest area (ha)'
    # output_flag = 0
    # output_folder = None
    # output_filename = None
    # title = None
    #
    # values_plot = sheet_pa.iloc[:, 6:8].values.T * 900 / 10000
    # pf_pa_stacked_plot(list_year,
    #                    values_plot,
    #                    x_label='Year',
    #                    y_label='Primary forest area (ha)',
    #                    title='Haiti: Primary wet forest',
    #                    output_flag=0,
    #                    output_folder=None,
    #                    output_filename=None,
    #                    legend_flag=True
    #                    )
    #
    # values_plot = sheet_pa.iloc[:, 9:11].values.T * 900 / 10000
    # pf_pa_stacked_plot(list_year,
    #                    values_plot,
    #                    x_label='Year',
    #                    y_label='Primary forest area (ha)',
    #                    title='Haiti: Primary dry forest',
    #                    output_flag=0,
    #                    output_folder=None,
    #                    output_filename=None,
    #                    )
    #
    # values_plot = sheet_pa.iloc[:, 15:17].values.T * 900 / 10000
    # pf_pa_stacked_plot(list_year,
    #                    values_plot,
    #                    x_label='Year',
    #                    y_label='Primary forest area (ha)',
    #                    title='Dominican Republic: Primary wet forest',
    #                    output_flag=0,
    #                    output_folder=None,
    #                    output_filename=None,
    #                    )
    #
    # values_plot = sheet_pa.iloc[:, 18:20].values.T * 900 / 10000
    # pf_pa_stacked_plot(list_year,
    #                    values_plot,
    #                    x_label='Year',
    #                    y_label='Primary forest area (ha)',
    #                    title='Dominican Republic: Primary dry forest',
    #                    output_flag=0,
    #                    output_folder=None,
    #                    output_filename=None,
    #                    )

    ##
    figure, axes = plt.subplots(ncols=2, nrows=2, figsize=(30, 16))

    ax = plt.subplot(2, 2, 1)
    values_plot = sheet_pa.iloc[:, 6:8].values.T * 900 / 1000 / 1000
    pf_pa_stacked_plot(ax,
                       list_year,
                       values_plot,
                       x_label='Year',
                       y_label='Primary forest area (km$\mathregular{^2}$)',
                       title='Haiti: Primary wet forest',
                       output_flag=0,
                       output_folder=None,
                       output_filename=None,
                       legend_flag=True
                       )

    ax = plt.subplot(2, 2, 2)
    values_plot = sheet_pa.iloc[:, 9:11].values.T * 900 / 1000 / 1000
    pf_pa_stacked_plot(ax,
                       list_year,
                       values_plot,
                       x_label='Year',
                       y_label='Primary forest area (km$\mathregular{^2}$)',
                       title='Haiti: Primary dry forest',
                       output_flag=0,
                       output_folder=None,
                       output_filename=None,
                       )

    ax = plt.subplot(2, 2, 3)
    values_plot = sheet_pa.iloc[:, 15:17].values.T * 900 / 1000 / 1000
    pf_pa_stacked_plot(ax,
                       list_year,
                       values_plot,
                       x_label='Year',
                       y_label='Primary forest area (km$\mathregular{^2}$)',
                       title='Dominican Republic: Primary wet forest',
                       output_flag=0,
                       output_folder=None,
                       output_filename=None,
                       )

    ax = plt.subplot(2, 2, 4)
    values_plot = sheet_pa.iloc[:, 18:20].values.T * 900 / 1000 / 1000
    pf_pa_stacked_plot(ax,
                       list_year,
                       values_plot,
                       x_label='Year',
                       y_label='Primary forest area (km$\mathregular{^2}$)',
                       title='Dominican Republic: Primary dry forest',
                       output_flag=0,
                       output_folder=None,
                       output_filename=None,
                       )

    plt.tight_layout()

    # plt.savefig(join(r'C:\Users\64937\OneDrive\LCM_biodiversity\manuscript\figure\protect_area_pf_change', '{}_protect_area_pf_change.jpg'.format(landcover_version)), dpi=300)