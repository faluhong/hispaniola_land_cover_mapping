"""
    plot the patch count, patch size for primary wet and dry forests in Haiti and Dominican Republic
"""

import numpy as np
import os
from os.path import join
import sys
import matplotlib.pyplot as plt
import pandas as pd
from osgeo import gdal_array
from scipy.ndimage import label, generate_binary_structure
import seaborn as sns
import matplotlib.ticker as plticker
import matplotlib

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

def subplot(ax1, array_year, pf_patch_count, pf_patch_size,
            x_axis_interval=3, y_axis_interval=None,
            title=None, plot_legend=False):

    legend_size = 22
    tick_label_size = 24
    axis_label_size = 27
    title_label_size = 30
    tick_length = 4
    lw = 2.5

    ax1.plot(array_year, pf_patch_count, label='patch number', color='#4c72b0', linewidth=lw, linestyle='solid',
             marker='o', markersize=9, markerfacecolor='#4c72b0',
             markeredgewidth=2, markeredgecolor='#4c72b0'
             )

    ax1.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    ax1.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=True, which='major', colors='#4c72b0')

    ax1.set_xlabel('year', size=axis_label_size)
    ax1.set_ylabel('patch number', size=axis_label_size, color='#4c72b0')

    if y_axis_interval is None:
        pass
    else:
        ax1.yaxis.set_major_locator(plticker.MultipleLocator(base=y_axis_interval))
    ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=x_axis_interval))

    ax1.yaxis.offsetText.set_fontsize(tick_label_size)
    ax1.set_title(title, fontsize=title_label_size)

    ax2 = ax1.twinx()
    ax2.plot(array_year, pf_patch_size, label='mean patch size', color='#ff4500',
             linewidth=lw, linestyle='solid',
             marker='s', markersize=8, markerfacecolor='#ff4500',
             markeredgewidth=2, markeredgecolor='#ff4500'
             )

    ax2.tick_params('x', labelsize=tick_label_size, direction='out', length=tick_length, bottom=True, which='major')
    ax2.tick_params('y', labelsize=tick_label_size, direction='out', length=tick_length, left=False, which='major', colors='#ff4500')

    ax2.set_ylabel('mean patch size', size=axis_label_size, color='#ff4500')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    if plot_legend:
        ax1.legend(lines + lines2, labels + labels2, loc='best', fontsize=legend_size, frameon=False)

    # ax1.grid(False)
    # ax1.yaxis.grid(False)  # Hide the horizontal gridlines
    # ax1.xaxis.grid(True)

# def main():
if __name__=='__main__':
    landcover_version = 'degrade_v2_refine_3_3'
    # landcover_version = 'irf_v59'

    filename = join(rootpath, 'results', '{}_landcover_classification'.format(landcover_version), 'pf_landscape_metrix.xlsx')

    df = pd.read_excel(filename)

    array_year = np.arange(1996, 2023)

    # sns.set_theme()
    sns.set_style('white')

    ##
    title = None
    y_label = 'patch number'
    x_axis_interval = 3
    y_axis_interval = None

    matplotlib.rcParams['font.family'] = 'Arial'
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(24, 12))

    for i in range(0, 2):
        for j in range(0, 2):

            if (i == 0) & (j == 0):
                ax1 = axes[i, j]

                pf_patch_count = df['value'].values[(df['country'] == 'haiti') & (df['pf_flag'] == 'wet') & (df['patch_flag'] == 'count')]
                pf_patch_size = df['value'].values[(df['country'] == 'haiti') & (df['pf_flag'] == 'wet') & (df['patch_flag'] == 'mean_size')]

                subplot(ax1, array_year, pf_patch_count, pf_patch_size, x_axis_interval=3, y_axis_interval=None,
                        title='Haiti: primary wet forest', plot_legend=True)

            elif (i == 0) & (j == 1):
                ax1 = axes[i, j]

                pf_patch_count = df['value'].values[(df['country'] == 'haiti') & (df['pf_flag'] == 'dry') & (df['patch_flag'] == 'count')]
                pf_patch_size = df['value'].values[(df['country'] == 'haiti') & (df['pf_flag'] == 'dry') & (df['patch_flag'] == 'mean_size')]

                subplot(ax1, array_year, pf_patch_count, pf_patch_size, x_axis_interval=3, y_axis_interval=None, title='Haiti: primary dry forest')

            elif (i == 1) & (j == 0):
                ax1 = axes[i, j]

                pf_patch_count = df['value'].values[(df['country'] == 'dr') & (df['pf_flag'] == 'wet') & (df['patch_flag'] == 'count')]
                pf_patch_size = df['value'].values[(df['country'] == 'dr') & (df['pf_flag'] == 'wet') & (df['patch_flag'] == 'mean_size')]

                subplot(ax1, array_year, pf_patch_count, pf_patch_size, x_axis_interval=3, y_axis_interval=None, title='Dominican Republic: primary wet forest')

            else:
                ax1 = axes[i, j]

                pf_patch_count = df['value'].values[(df['country'] == 'dr') & (df['pf_flag'] == 'dry') & (df['patch_flag'] == 'count')]
                pf_patch_size = df['value'].values[(df['country'] == 'dr') & (df['pf_flag'] == 'dry') & (df['patch_flag'] == 'mean_size')]

                subplot(ax1, array_year, pf_patch_count, pf_patch_size, x_axis_interval=3, y_axis_interval=None, title='Dominican Republic: primary dry forest')

    plt.tight_layout()

    plt.savefig(r'C:\Users\64937\OneDrive\LCM_biodiversity\manuscript\figure\fragmentation\pf_patch_count_size.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    ##

