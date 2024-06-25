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
# import warnings
# warnings.filterwarnings("ignore")

# from accuracy_assessment.generate_validation_sample_pf_loss import get_pf_loss_map
# from accuracy_assessment.val_report_mutual_pf_loss_sample import validation_pf_loss_summary
# from accuracy_assessment.validation_matrix import plot_df_confusion
# from accuracy_assessment.calulate_validation_report_land_cover import get_count_weight_merge_barren_cropland
# from accuracy_assessment.val_report_first_round_lc import summary_accuracy_evaluation

from accuracy_assessment.good_practice_accuracy_assessment import generate_good_practice_matrix

# def main():
if __name__ == '__main__':

    pwd = os.getcwd()
    rootpath = os.path.abspath(os.path.join(pwd, '..'))

    landcover_system = {'1': 'developed',
                        '2': 'primary wet forest',
                        '3': 'primary dry forest',
                        '4': 'secondary forest',
                        '5': 'shrub/grass',
                        '6': 'water',
                        '7': 'wetland',
                        '8': 'barren/cropland'}
    reverse_lc_system = {v: int(k) for k, v in landcover_system.items()}

    pf_loss_dict = {'1': 'Non PF loss',
                    '2': 'PF loss'}

    filename_lc_pct = join(rootpath, 'results', 'land_cover_pct.xlsx')
    df_lc_pct = pd.read_excel(filename_lc_pct, sheet_name='Hispaniola')

    array_count_lc = df_lc_pct.iloc[:, 2:10].sum().values
    array_weight_lc = array_count_lc / np.nansum(array_count_lc)


    filename_accuracy_assessment_tabel = join(rootpath, 'results', 'accuracy_assessment_table.xlsx')

    df_lc_assessment = pd.read_excel(filename_accuracy_assessment_tabel, sheet_name='landcover_validation_record')

    array_lc_map = df_lc_assessment['map_type'].values
    array_lc_map = np.array([reverse_lc_system.get(i, -999) for i in array_lc_map])

    exclude_flag = df_lc_assessment['final_exclude_flag'].values == True
    array_lc_reference = df_lc_assessment['final_lc'].values
    array_lc_reference = np.array([reverse_lc_system.get(i, -999) for i in array_lc_reference])

    ##
    mask_exclude = (array_lc_map == -999) | (array_lc_reference == -999) | (exclude_flag)


    array_lc_map_final = array_lc_map[~mask_exclude]
    array_lc_reference_final = array_lc_reference[~mask_exclude]

    categories = np.arange(1, len(array_weight_lc) + 1)

    array_map = pd.Categorical(array_lc_map_final, categories=categories)  # define the categories to avoid missing categories in the confusion matrix
    array_primary_validation = pd.Categorical(array_lc_reference_final, categories=categories)

    df_confusion = pd.crosstab(array_map, array_primary_validation, rownames=['Map'], colnames=['Validation'], dropna=False)
    overall_accuracy = np.trace(df_confusion.values) / np.sum(df_confusion.values)

    print(f'number of agreement pixels: {np.trace(df_confusion.values)} / {df_confusion.values.sum()}')
    print(f'count-based overall accuracy {overall_accuracy}')

    df_err_adjust = generate_good_practice_matrix(df_confusion.values, array_weight_lc, array_count_lc)
    print('adjusted overall accuracy {}'.format(df_err_adjust.loc['PA', 'UA']))

    ##

    # filename_collection = join(rootpath, 'results', 'accuracy_assessment', 'degrade_v2_refine_3_3_separate_validation',
    #                            're_interpretation_sample',
    #                            'Collection_re_interpretation_sample.xlsx')
    #
    # sheet_lc_validation = pd.read_excel(filename_collection, sheet_name='land_cover_reinterpretation')
    # index_lc = sheet_lc_validation['Index'].values
    # exclude_flag_lc = sheet_lc_validation['final_exclude_flag'].values == True
    #
    # array_lc = sheet_lc_validation['final_pri_lc'].values
    # array_lc = np.array([reverse_lc_system.get(i, -999) for i in array_lc])
    #
    # sheet_pf_loss_validation = pd.read_excel(filename_collection, sheet_name='pf_loss_reinterpretation')
    # index_pf_loss = sheet_pf_loss_validation['Index'].values
    # exclude_flag_pf_loss = sheet_pf_loss_validation['final_exclude_flag'].values == True
    #
    # array_pf_loss = sheet_pf_loss_validation['final_val_label'].values
    # array_pf_loss = np.array([reverse_pf_loss_system.get(i, -999) for i in array_pf_loss])

    ##

    # mask_exclude_lc = np.ones(len(array_lc_first_round_validation), dtype=bool)
    # mask_exclude_lc[(array_lc_map == -999) | (array_lc_all_new == -999) | (array_lc_all_new == 999)] = False
    # mask_exclude_lc[array_lc_first_round_exclude] = False
    #
    # df_confusion_lc, df_err_adjust_lc = summary_accuracy_evaluation(array_lc_map, array_lc_all_new, mask_exclude_lc,
    #                                                                 array_weight_lc, array_counts_lc)
    # plot_df_confusion(df_confusion_lc.values, landcover_system=landcover_system, title='final-round-reinterpretation', figsize=(11.5, 8))
    #
    # ##
    #
    # array_pf_loss_all_new = array_pf_loss_first_round_validation.copy()
    # array_pf_loss_all_new[index_pf_loss - 1] = array_pf_loss
    #
    # mask_exclude = np.ones(len(array_pf_loss_first_round_validation), dtype=bool)
    # mask_exclude[(array_pf_loss_first_round_validation == -999) | (array_pf_loss_all_new == -999) | (array_pf_loss_all_new == 999)] = False
    # mask_exclude[array_pf_loss_first_round_exclude] = False
    #
    # df_confusion_pf_loss, df_err_adjust_pf_loss, f1_score_pf_loss = validation_pf_loss_summary(array_pf_loss_map[mask_exclude], array_pf_loss_all_new[mask_exclude],
    #                                                                            array_weight_pf_loss, array_counts_pf_loss)
    # plot_df_confusion(df_confusion_pf_loss.values, landcover_system=pf_loss_dict, title='final-round-reinterpretation', figsize=(11.5, 8))

