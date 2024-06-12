"""
plot the percentage change of each land cover, but split the large and small percentage ones for better visualization
"""
import matplotlib
# matplotlib.use('qtagg')
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg',force=True)
import numpy as np
from os.path import join
import pandas as pd
import matplotlib.ticker as plticker
import seaborn as sns
import os
import sys
import matplotlib.ticker as ticker

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

def format_year(x, pos=None):
    return str(int(x))[-2:]

def plot_pct_change_nine_split(list_year_plot,
                               df_observe,
                               title=None,
                               ax_1_interval=5.0,
                               ax_2_interval=1.0,
                               output_flag=0,
                               output_folder=' ',
                               output_filename=' '):
    """
        plot the area and percentage change of each land cover, but split the large and small percentage ones for better visualization
        The left y-axis is the area, the right y-axis is the percentage
    """

    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(16, 15))
    legend_size = 20
    tick_label_size = 30
    axis_label_size = 36
    title_size = 40
    tick_length = 4

    linewidth = 3.0
    linestyle = 'solid'

    for axis in ['top', 'bottom', 'left', 'right']:
        axes[0].spines[axis].set_linewidth(2)
        axes[1].spines[axis].set_linewidth(2)

    axes[0].set_title(title, fontsize=title_size)

    colors = np.array([np.array([241, 1, 0, 255]) / 255,
                       np.array([179, 175, 164, 255]) / 255,
                       np.array([29, 101, 51, 255]) / 255,
                       np.array([244, 127, 17, 255]) / 255,
                       np.array([108, 169, 102, 255]) / 255,
                       np.array([208, 209, 129, 255]) / 255,  # FFA500
                       np.array([174, 114, 41, 255]) / 255,
                       np.array([72, 109, 162, 255]) / 255,
                       np.array([186, 85, 211, 255]) / 255
                       ])

    # axes[0].plot(list_year_plot, df_observe['2 Barren %'].values * 100, label='Barren', color=colors[1],
    #              linestyle=linestyle, linewidth=linewidth)
    # axes[0].plot(list_year_plot, df_observe['5 Secondary forest %'].values * 100, label='Secondary forest',
    #              color=colors[4], linestyle=linestyle, linewidth=linewidth)
    # axes[0].plot(list_year_plot, df_observe['7 Cropland %'].values * 100, label='Cropland', color=colors[6],
    #              linestyle=linestyle, linewidth=linewidth)
    #
    # axes[1].plot(list_year_plot, df_observe['1 Developed %'].values * 100, label='Developed', color=colors[0],
    #              linestyle=linestyle, linewidth=linewidth)
    # axes[1].plot(list_year_plot, df_observe['3 Primary wet forest %'].values * 100, label='Primary wet forest', color=colors[2],
    #              linestyle=linestyle, linewidth=linewidth)
    # axes[1].plot(list_year_plot, df_observe['4 Primary dry forest %'].values * 100, label='Primary dry forest', color=colors[3],
    #              linestyle=linestyle, linewidth=linewidth)
    # axes[1].plot(list_year_plot, df_observe['6 Shrub/Grass %'].values * 100, label='Shrub/Grass', color=colors[5],
    #              linestyle=linestyle, linewidth=linewidth)
    # axes[1].plot(list_year_plot, df_observe['8 Water %'].values * 100, label='Water', color=colors[7],
    #              linestyle=linestyle, linewidth=linewidth)
    # axes[1].plot(list_year_plot, df_observe['9 Wetland %'].values * 100, label='Wetland', color=colors[8],
    #              linestyle=linestyle, linewidth=linewidth)

    # axes[0].plot(list_year_plot, df_observe['2 Barren'].values * 900 / 1000000, label='Barren', color=colors[1],
    #              linestyle=linestyle, linewidth=linewidth)
    # axes[0].plot(list_year_plot, df_observe['5 Secondary forest'].values * 900 / 1000000, label='Secondary forest',
    #              color=colors[4], linestyle=linestyle, linewidth=linewidth)
    # axes[0].plot(list_year_plot, df_observe['7 Cropland'].values * 900 / 1000000, label='Cropland', color=colors[6],
    #              linestyle=linestyle, linewidth=linewidth)

    axes[0].plot(list_year_plot, (df_observe['2 Barren'].values + df_observe['7 Cropland'].values) * 900 / 1000000, label='Other', color=colors[1],
                 linestyle=linestyle, linewidth=linewidth)
    axes[0].plot(list_year_plot, df_observe['5 Secondary forest'].values * 900 / 1000000, label='Secondary forest',
                 color=colors[4], linestyle=linestyle, linewidth=linewidth)
    # axes[0].plot(list_year_plot, df_observe['7 Cropland'].values * 900 / 1000000, label='Cropland', color=colors[6],
    #              linestyle=linestyle, linewidth=linewidth)

    axes[1].plot(list_year_plot, df_observe['1 Developed'].values * 900 / 1000000, label='Developed', color=colors[0],
                 linestyle=linestyle, linewidth=linewidth)
    axes[1].plot(list_year_plot, df_observe['3 Primary wet forest'].values * 900 / 1000000, label='Primary wet forest', color=colors[2],
                 linestyle=linestyle, linewidth=linewidth)
    axes[1].plot(list_year_plot, df_observe['4 Primary dry forest'].values * 900 / 1000000, label='Primary dry forest', color=colors[5],
                 linestyle=linestyle, linewidth=linewidth)
    axes[1].plot(list_year_plot, df_observe['6 Shrub/Grass'].values * 900 / 1000000, label='Shrub/Grass', color=colors[6],
                 linestyle=linestyle, linewidth=linewidth)
    axes[1].plot(list_year_plot, df_observe['8 Water'].values * 900 / 1000000, label='Water', color=colors[7],
                 linestyle=linestyle, linewidth=linewidth)
    axes[1].plot(list_year_plot, df_observe['9 Wetland'].values * 900 / 1000000, label='Wetland', color=colors[8],
                 linestyle=linestyle, linewidth=linewidth)

    for i_ax in range(0, 2):
        ax_tmp = axes[i_ax]

        ax_tmp.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
        ax_tmp.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major')

        ax2 = ax_tmp.secondary_yaxis('right')   # set the second y-axis, copy from the left y-axis
        ax2.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
        ax2.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, right=True, which='major')

        if i_ax == 0:
            ax_tmp.yaxis.set_major_locator(plticker.MultipleLocator(base=ax_1_interval))
            ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=ax_1_interval))
        else:
            ax_tmp.yaxis.set_major_locator(plticker.MultipleLocator(base=ax_2_interval))
            ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=ax_2_interval))

        if i_ax == 1:
            ax_tmp.set_xlabel('year', size=axis_label_size)

        ax_ticks = ax_tmp.get_yticks()    # get the ticks of the left y-axis
        ax_lables = ax_ticks / (df_observe['TOTAL'].values[-1] * 900 / 1000000) * 100    # convert the area to percentage
        ax_lables = np.round(ax_lables, 1)
        ax2.set_yticklabels(ax_lables)   # set the right y-axis with the percentage label

        ax_tmp.set_ylabel('Area (km$^2$)', size=axis_label_size)
        ax2.set_ylabel('Percentage (%)', size=axis_label_size, labelpad=15)

        # Create a FuncFormatter object using the format_year function, change the four-digit year to two-digit year
        year_formatter = ticker.FuncFormatter(format_year)
        ax_tmp.xaxis.set_major_formatter(year_formatter)

        ax_tmp.xaxis.set_major_locator(plticker.MultipleLocator(base=2.0))

    if legend_flag:
        lines_1, labels_1 = axes[0].get_legend_handles_labels()
        lines_2, labels_2 = axes[1].get_legend_handles_labels()

        legend_lines = [lines_2[0], lines_1[0], lines_2[1], lines_2[2], lines_1[1], lines_2[3], lines_1[2], lines_2[4], lines_2[5]]
        legend_label = [labels_2[0], labels_1[0], labels_2[1], labels_2[2], labels_1[1], labels_2[3], labels_1[2], labels_2[4], labels_2[5]]

        axes[0].legend(legend_lines, legend_label, loc='best', fontsize=legend_size, bbox_to_anchor=(1.01, 1))

    plt.tight_layout()

    if output_flag == 0:
        plt.show()
    else:
        output_file = join(output_folder, '{}.jpg'.format(output_filename))
        plt.savefig(output_file, dpi=300)
        plt.close()


# def main():
if __name__ == '__main__':

    # output_version_flag = 'irf_v59'
    output_version_flag = 'degrade_v2_refine_3_3'

    # list_year = np.arange(1984, 2023)
    np.set_printoptions(precision=4, suppress=True)

    filename_percentile = join(rootpath, 'results', '{}_landcover_classification'.format(output_version_flag),
                               '{}_landcover_analysis.xlsx'.format(output_version_flag))

    sheet_hispaniola = pd.read_excel(filename_percentile, sheet_name='Hispaniola')
    sheet_haiti = pd.read_excel(filename_percentile, sheet_name='Haiti')
    sheet_dr = pd.read_excel(filename_percentile, sheet_name='Dominican')

    list_year_plot = np.arange(1996, 2023)
    sheet_hispaniola= sheet_hispaniola[sheet_hispaniola['Year'].isin(list_year_plot)]
    sheet_haiti = sheet_haiti[sheet_haiti['Year'].isin(list_year_plot)]
    sheet_dr = sheet_dr[sheet_dr['Year'].isin(list_year_plot)]

    ##

    sns.set_style("white")

    # plot_pct_change_nine_split(list_year_plot,
    #                       sheet_haiti,
    #                       title=' ',
    #                       ax_1_interval=5.0,
    #                       ax_2_interval=0.5,
    #                       output_flag=0, output_folder=' ')
    #
    # plot_pct_change_nine_split(list_year_plot,
    #                       sheet_dr,
    #                       title=' ',
    #                       ax_1_interval=5.0,
    #                       ax_2_interval=1.0,
    #                       output_flag=0, output_folder=' ')

    ##

    df_observe = sheet_haiti
    title = ' '
    ax_1_interval = 2000
    ax_2_interval = 100
    output_flag = 1
    output_folder = ' '
    legend_flag = False

    ##
    output_folder = r'C:\Users\64937\OneDrive\LCM_biodiversity\manuscript\figure\landcover_pct_change'
    plot_pct_change_nine_split(list_year_plot,
                               sheet_haiti,
                               title=' ',
                               ax_1_interval=2000,
                               ax_2_interval=100,
                               output_flag=1,
                               output_folder=output_folder,
                               output_filename='Haiti_LC_pct_merged_8types')

    plot_pct_change_nine_split(list_year_plot,
                               sheet_dr,
                               title=' ',
                               ax_1_interval=500,
                               ax_2_interval=400,
                               output_flag=1,
                               output_folder=output_folder,
                               output_filename='DR_LC_pct_merged_8types')

    ##

