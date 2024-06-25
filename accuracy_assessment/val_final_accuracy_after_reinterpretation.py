"""
    Report the land cover and PF loss accuracy_assessment results after the re-interpretation and group discussion
"""

import time
import numpy as np
import numpy.typing as npt
import os
from os.path import join
import sys
from osgeo import gdal, gdal_array
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import fiona
import geopandas as gpd
import seaborn as sns

from accuracy_assessment.good_practice_accuracy_assessment import (generate_good_practice_matrix,
                                                                   plot_df_confusion, get_adjusted_area_and_margin_of_error)

# def main():
if __name__ == '__main__':

    pwd = os.getcwd()
    rootpath = os.path.abspath(os.path.join(pwd, '..'))

    filename_accuracy_assessment_tabel = join(rootpath, 'results', 'accuracy_assessment_table.xlsx')

    landcover_system = {'1': 'developed',
                        '2': 'primary wet forest',
                        '3': 'primary dry forest',
                        '4': 'secondary forest',
                        '5': 'shrub/grass',
                        '6': 'water',
                        '7': 'wetland',
                        '8': 'barren/cropland'}
    reverse_lc_system = {v: int(k) for k, v in landcover_system.items()}

    filename_lc_pct = join(rootpath, 'results', 'land_cover_pct.xlsx')
    df_lc_pct = pd.read_excel(filename_lc_pct, sheet_name='Hispaniola')

    # get the count and weight
    array_count_lc = df_lc_pct.iloc[:, 2:10].sum().values
    array_weight_lc = array_count_lc / np.nansum(array_count_lc)

    # read the land cover assessment data
    df_lc_assessment = pd.read_excel(filename_accuracy_assessment_tabel, sheet_name='landcover_validation_record')

    # get the map results
    array_lc_map = df_lc_assessment['map_type'].values
    array_lc_map = np.array([reverse_lc_system.get(i, -999) for i in array_lc_map])

    # get the reference results
    exclude_flag = df_lc_assessment['final_exclude_flag'].values == True
    array_lc_reference = df_lc_assessment['final_lc'].values
    array_lc_reference = np.array([reverse_lc_system.get(i, -999) for i in array_lc_reference])

    # final data to report the accuracy
    mask_exclude = (array_lc_map == -999) | (array_lc_reference == -999) | exclude_flag

    array_lc_map_final = array_lc_map[~mask_exclude]
    array_lc_reference_final = array_lc_reference[~mask_exclude]

    # count-based confusion matrix
    categories = np.arange(1, len(array_weight_lc) + 1)

    array_lc_map_final = pd.Categorical(array_lc_map_final, categories=categories)  # define the categories to avoid missing categories in the confusion matrix
    array_lc_reference_final = pd.Categorical(array_lc_reference_final, categories=categories)

    df_confusion_lc = pd.crosstab(array_lc_map_final, array_lc_reference_final, rownames=['Map'], colnames=['Reference'], dropna=False)
    overall_accuracy = np.trace(df_confusion_lc.values) / np.sum(df_confusion_lc.values)

    print(f'number of agreement pixels: {np.trace(df_confusion_lc.values)} / {df_confusion_lc.values.sum()}')
    print(f'count-based overall accuracy {overall_accuracy}')

    plot_df_confusion(df_confusion_lc.values, stratum_des=landcover_system, title='land cover', figsize=(11.5, 8))

    # area-based confusion matrix
    df_err_adjust_lc = generate_good_practice_matrix(df_confusion_lc.values, array_weight_lc, array_count_lc)
    print('adjusted overall accuracy {}'.format(df_err_adjust_lc.loc['PA', 'UA']))

    # get area adjustment results and margin of error
    get_adjusted_area_and_margin_of_error(df_confusion_lc.values, df_err_adjust_lc, array_count_lc, confidence_interval=1.96)

    ##
    pf_loss_dict = {'1': 'Non PF loss',
                    '2': 'PF loss'}
    reverse_pf_loss_system = {v: int(k) for k, v in pf_loss_dict.items()}

    # get the count and weight
    count_pf_loss = df_lc_pct.iloc[0, 3:5].sum() - df_lc_pct.iloc[-1, 3:5].sum()
    count_other = df_lc_pct.loc[0, 'TOTAL'] - count_pf_loss

    array_count_pf_loss = np.array([count_other, count_pf_loss])
    array_weight_pf_loss = array_count_pf_loss / np.nansum(array_count_pf_loss)

    # read the PF loss assessment data
    df_pf_loss_assessment = pd.read_excel(filename_accuracy_assessment_tabel, sheet_name='pf_loss_validation_record')

    # get the map results
    array_pf_loss_map = df_pf_loss_assessment['map_type'].values
    array_pf_loss_map = np.array([reverse_pf_loss_system.get(i, -999) for i in array_pf_loss_map])

    # get the reference results
    exclude_flag_pf_loss = df_pf_loss_assessment['final_exclude_flag'].values == True
    array_pf_loss_reference = df_pf_loss_assessment['final_val_label'].values
    array_pf_loss_reference = np.array([reverse_pf_loss_system.get(i, -999) for i in array_pf_loss_reference])

    # final data to report the accuracy
    mask_exclude_pf_loss = (array_pf_loss_map == -999) | (array_pf_loss_reference == -999) | exclude_flag_pf_loss
    array_pf_loss_map_final = array_pf_loss_map[~mask_exclude_pf_loss]
    array_pf_loss_reference_final = array_pf_loss_reference[~mask_exclude_pf_loss]

    # count-based confusion matrix
    categories = np.array([1, 2])

    array_pf_loss_map_final = pd.Categorical(array_pf_loss_map_final, categories=categories)  # define the categories to avoid missing categories in the confusion matrix
    array_pf_loss_reference_final = pd.Categorical(array_pf_loss_reference_final, categories=categories)

    df_confusion_pf_loss = pd.crosstab(array_pf_loss_map_final, array_pf_loss_reference_final, rownames=['Map'], colnames=['Reference'], dropna=False)
    overall_accuracy = np.trace(df_confusion_pf_loss.values) / np.sum(df_confusion_pf_loss.values)

    print(f'number of agreement pixels: {np.trace(df_confusion_pf_loss.values)} / {df_confusion_pf_loss.values.sum()}')
    print(f'count-based overall accuracy {overall_accuracy}')

    plot_df_confusion(df_confusion_pf_loss.values, stratum_des=pf_loss_dict, title='PF loss', figsize=(11.5, 8))

    # area-based confusion matrix
    df_err_adjust_pf_loss = generate_good_practice_matrix(df_confusion_pf_loss.values, array_weight_pf_loss, array_count_pf_loss)
    print('adjusted overall accuracy {}'.format(df_err_adjust_pf_loss.loc['PA', 'UA']))

    ua_pf_loss = df_err_adjust_pf_loss.loc[2, 'UA']
    pa_pf_loss = df_err_adjust_pf_loss.loc['PA', 2]
    print('user accuracy for PF loss: {}'.format(ua_pf_loss))
    print('producer accuracy for PF loss: {}'.format(pa_pf_loss))

    ##
    get_adjusted_area_and_margin_of_error(df_confusion_pf_loss.values, df_err_adjust_pf_loss, array_count_pf_loss, confidence_interval=1.96)

