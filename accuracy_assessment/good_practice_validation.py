"""
    generate the accuracy_assessment matrix recommended by Olofsson et al. (2014) following the "Good Practice" strategy
"""

import time
import numpy as np
import os
from os.path import join
import sys
from osgeo import gdal, gdal_array
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import earthpy.plot as ep
from sklearn.metrics import accuracy_score
import seaborn as sn

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

# from Basic_tools.Figure_plot import FP_ISP, FP, band_plot, RGB_composite, qc_plot, fp_imshow
# from Landcover_classification.landcover_function import land_cover_plot
# from Basic_tools.Error_statistical import Error_statistical

def plot_df_confusion(array, landcover_system=None, figsize=(16, 13), title=None, x_label='Validation', y_label='Map'):
    """
        plot the confusion matrix
    """
    # array = df_confusion.values

    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'Barren',
                            '3': 'PrimaryForest',
                            '4': 'SecondaryForest',
                            '5': 'ShrubGrass',
                            '6': 'Cropland',
                            '7': 'Water',
                            '8': 'Wetland'}

    df_cm = pd.DataFrame(array, index=landcover_system.values(), columns=landcover_system.values())

    figure, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    cmap = matplotlib.cm.GnBu

    tick_labelsize = 16
    axis_labelsize = 20
    title_size = 24
    annosize = 18
    ticklength = 4
    axes_linewidth = 1.5

    im = sn.heatmap(df_cm, annot=True, annot_kws={"size": annosize}, fmt='d', cmap=cmap, cbar=False)
    im.figure.axes[-1].yaxis.set_tick_params(labelsize=annosize)

    ax.tick_params('y', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, left=True, which='major', rotation=0)
    ax.tick_params('x', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, top=False, which='major', rotation=15)

    ax.set_xlabel(x_label, size=axis_labelsize)
    ax.set_ylabel(y_label, size=axis_labelsize)

    overall_accuracy = np.trace(df_cm.values) / np.sum(df_cm.values)
    # print(f'number of agreement pixels: {np.trace(df_cm.values)} / {df_cm.values.sum()}')
    # print(f'count-based overall accuracy {overall_accuracy}')

    ax.set_title(f'{title}:  {np.trace(df_cm.values)}/{df_cm.values.sum()}={np.round(overall_accuracy*100, 1)}', size=title_size)

    plt.tight_layout()


def get_lc_weight(landcover_version):
    """
        get the weight of each land cover type
    """

    filename_landcover_statis = join(rootpath, 'results', '{}_landcover_classification'.format(landcover_version),
                                     '{}_landcover_analysis.xlsx'.format(landcover_version))
    sheet_landcover = pd.read_excel(filename_landcover_statis)

    array_count = sheet_landcover.iloc[12::, 2:11].sum().values
    array_weight = array_count[0:-1] / array_count[-1]

    return array_weight, array_count[0:-1]


def generate_good_practice_matrix(data, array_weight, array_count, confidence_interval=1.96):
    """
        generate the error matrix following the "Good Practice" strategy
        The accuracy_assessment sample design follows the stratified random sample

        Steps:
        (1) calculate the proportion of area for each cell
        (2) calculate the user's and producer's accuracy
        (3) calculate the variance of user's, producer's and overall accuracy
        (4) reorder the output dataframe

        Ref: Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014).
        Good practices for estimating area and assessing accuracy of land change. Remote sensing of Environment, 148, 42-57.
        https://www.sciencedirect.com/science/article/pii/S0034425714000704

        Args:
            data: 2-D array, the error matrix of sample counts
                  ROW represents the map class, COLUMN represents the reference class
            array_weight: weight of each class
            array_count: pixel count of each class
            confidence_interval: value representing the confidence interval, default is 1.96, indicating the 5-95% interval
        Returns:
            df_error_adjust: error matrix based on the proportions of area
    """

    landcover_types = len(array_weight)
    list_landcover = list(np.arange(1, 1 + landcover_types))  # list to indicate the land cover types

    df_confusion = pd.DataFrame(data=data, columns=list_landcover, index=list_landcover)

    df_err_adjust = pd.DataFrame(df_confusion.iloc[0:landcover_types, 0:landcover_types], index=list_landcover, columns=list_landcover)

    df_err_adjust.loc[:, 'n_count'] = df_err_adjust.sum(axis=1)
    df_err_adjust.loc['n_count', :] = df_err_adjust.sum(axis=0)

    df_err_adjust.loc[:, 'UA'] = np.nan
    df_err_adjust.loc['PA', :] = np.nan

    df_err_adjust.loc[list_landcover, 'weight'] = array_weight  # assign weight

    for i_row in range(0, landcover_types):
        df_err_adjust.iloc[i_row, 0: landcover_types] = df_err_adjust.iloc[i_row, 0: landcover_types] / \
                                                        df_err_adjust.loc[i_row + 1, 'n_count'] * df_err_adjust.loc[i_row + 1, 'weight']
        df_err_adjust.loc[i_row + 1, 'total'] = np.nansum(df_err_adjust.iloc[i_row, 0: landcover_types])

    for i_col in range(0, landcover_types):
        df_err_adjust.loc['total', i_col + 1] = np.nansum(df_err_adjust.iloc[0: landcover_types, i_col])

    # calculate the user's and producer's accuracy
    for i in range(0, landcover_types):
        df_err_adjust.loc['PA', i + 1] = df_err_adjust.iloc[i, i] / df_err_adjust.loc['total', i + 1]
        df_err_adjust.loc[i + 1, 'UA'] = df_err_adjust.iloc[i, i] / df_err_adjust.loc[i + 1, 'total']

    df_err_adjust.loc['total', 'total'] = 1

    # calculate the overall accuracy
    overall_accuracy = np.nansum(np.diag(df_err_adjust.iloc[0: landcover_types, 0: landcover_types].values))
    df_err_adjust.loc['PA', 'UA'] = overall_accuracy

    # calculate the user's accuracy variance
    user_accuracy = df_err_adjust['UA'].values[0: landcover_types]
    variance_user_accuracy = user_accuracy * (1 - user_accuracy) / (df_err_adjust['n_count'].values[0: landcover_types] - 1)
    std_user_accuracy = np.sqrt(variance_user_accuracy)

    # calculate the overall accuracy variance
    variance_overall_accuracy = np.power(array_weight, 2) * user_accuracy * (1 - user_accuracy) / (df_err_adjust['n_count'].values[0: landcover_types] - 1)
    variance_overall_accuracy = np.sum(variance_overall_accuracy)
    std_overall_accuracy = np.sqrt(variance_overall_accuracy)

    # calculate the producer's accuracy variance
    producer_accuracy = df_err_adjust.loc['PA', :][0: landcover_types].values
    variance_producer_accuracy = np.zeros(np.shape(producer_accuracy), dtype=float)

    for j in range(0, landcover_types):

        # calculate the estimated Nj
        N_j_estimated = 0
        for i in range(0, landcover_types):
            N_j_estimated = N_j_estimated + array_count[i] / df_confusion.sum(axis=1).values[i] * df_confusion.iloc[i, j]

        # part 1
        part_1 = np.power(array_count[j] * (1 - producer_accuracy[j]), 2) * user_accuracy[j] * (1 - user_accuracy[j])
        n_j = df_confusion.sum(axis=1).values[j]
        part_1 = part_1 / (n_j - 1)

        # part 2
        part_2 = 0
        for i in range(0, landcover_types):
            if i == j:
                pass
            else:
                n_i = df_confusion.sum(axis=1).values[i]
                n_ij = df_confusion.iloc[i, j]

                tmp_1 = np.power(array_count[i], 2) / n_i * n_ij
                tmp_2 = (1 - n_ij / n_i) / (n_i - 1)

                part_2 = part_2 + tmp_1 * tmp_2

        part_2 = np.power(producer_accuracy[j], 2) * part_2

        # final variance producer accuracy
        variance_producer_accuracy[j] = (part_1 + part_2) / np.power(N_j_estimated, 2)

    # standard deviation of producer's accuracy
    std_producer_accuracy = np.sqrt(variance_producer_accuracy)

    # assign the uncertainty to the output table
    df_err_adjust.loc[:, 'UA_uncertainty'] = np.nan
    df_err_adjust['UA_uncertainty'].values[0: landcover_types] = std_user_accuracy * confidence_interval

    df_err_adjust.loc['PA_uncertainty', :] = np.nan
    # df_err_adjust.loc['PA_uncertainty', :].values[0: landcover_types] = std_producer_accuracy * confidence_interval    # No real data is assigned, this is wrong
    df_err_adjust.loc['PA_uncertainty', 1: landcover_types] = std_producer_accuracy * confidence_interval

    df_err_adjust.loc['PA_uncertainty', 'UA_uncertainty'] = std_overall_accuracy * confidence_interval

    # reorder the row and columns
    df_err_adjust = df_err_adjust[list_landcover + ['total', 'UA', 'UA_uncertainty', 'n_count', 'weight']]
    df_err_adjust = df_err_adjust.reindex(list_landcover + ['total', 'PA', 'PA_uncertainty', 'n_count'])

    return df_err_adjust


# def main():
if __name__=='__main__':

    landcover_version = 'irf_v52_0_5'

    array_weight, array_count = get_lc_weight(landcover_version)

    filename_validation = join(rootpath, 'results', 'accuracy_assessment', landcover_version, 'validation_sample_v1_record.xlsx')
    sheet_validation = pd.read_excel(filename_validation)
    sheet_validation = sheet_validation[~np.isnan(sheet_validation['pri_lc'])]

    primary_validation_lc = sheet_validation['pri_lc'].values
    map_lc = sheet_validation['lc_id'].values

    ##
    y_actu = pd.Series(primary_validation_lc, name='Validation')
    y_pred = pd.Series(map_lc, name='Map')
    df_confusion = pd.crosstab(y_pred, y_actu)

    # print('overall accuarcy {}'.format(accuracy_score(y_actu, y_pred)))
    # plot_df_confusion(df_confusion)
    ##
    df_err_adjust = generate_good_practice_matrix(df_confusion.values, array_weight, array_count)

    print('number of interpreted sample is {}'.format(df_err_adjust.loc['n_count', 'n_count']))
    print('overall accuracy is {}'.format(df_err_adjust.loc['PA', 'UA']))

    ## output the error adjust file
    # output_filename = join(rootpath, 'results', 'accuracy_assessment', landcover_version, 'adjust_validation_matrix.xlsx')
    # output_excel = pd.ExcelWriter(output_filename)
    #
    # df_confusion.to_excel(output_excel, sheet_name='pixel_count')
    # df_err_adjust.to_excel(output_excel, sheet_name='error_adjust')
    #
    # output_excel.close()
    ## plot the accuracy_assessment confusion matrix
    # from pretty_confusion_matrix import pp_matrix
    #
    # df_cm = pd.DataFrame(array, index=landcover_system.values(), columns=landcover_system.values())
    # # colormap: see this and choose your more dear
    # cmap = matplotlib.cm.GnBu
    # pp_matrix(df_cm, cmap=cmap)
