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
import warnings
warnings.filterwarnings("ignore")

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from accuracy_assessment.generate_validation_sample_pf_loss import get_pf_loss_map
from accuracy_assessment.val_report_mutual_pf_loss_sample import validation_pf_loss_summary
from accuracy_assessment.validation_matrix import plot_df_confusion
from accuracy_assessment.calulate_validation_report_land_cover import get_count_weight_merge_barren_cropland
from accuracy_assessment.val_report_first_round_lc import summary_accuracy_evaluation


def prepare_pf_loss_data():
    """
        Prepare the PF loss accuracy_assessment data, including
        (1) The weight and count of PF loss and Non PF loss types
        (2) The PF loss map
        (3) The first-round interpretation results
        (4) The exclude flag
    """

    pf_loss_dict = {'1': 'Non PF loss',
                    '2': 'PF loss'}
    reverse_validation_system = {v: int(k) for k, v in pf_loss_dict.items()}

    path_validation = join(rootpath, 'results', 'accuracy_assessment', 'degrade_v2_refine_3_3_separate_validation','pf_loss_final_sample')

    change_map_pf_loss = get_pf_loss_map(landcover_version='degrade_v2_refine_3_3')

    array_unique_values, array_counts = np.unique(change_map_pf_loss, return_counts=True)
    array_weight = array_counts[1::] / array_counts[1::].sum()

    # get the map types
    filename_map = join(path_validation, 'validation_pf_loss_400_python_output_fixed_for_share.xlsx')
    sheet_map = pd.read_excel(filename_map, sheet_name='Sheet1')
    array_map = sheet_map['map_type'].values

    array_map = np.array([reverse_validation_system.get(i, -999) for i in array_map])

    filename_validation = join(path_validation, 'collection_validation_pf_loss_400samples.xlsx')
    df_interpretation = pd.read_excel(filename_validation, sheet_name='validation_record')

    array_pf_loss_validation = df_interpretation['val_label'].values
    array_pf_loss_validation = np.array([reverse_validation_system.get(i, -999) for i in array_pf_loss_validation])

    array_exclude = df_interpretation['exclude_flag'].values == True

    return array_map, array_weight, array_counts, array_pf_loss_validation, array_exclude, reverse_validation_system


def prepare_lc_map_data():
    """
        Prepare the land cover accuracy_assessment data, including
        (1) The weight and count of each land cover type
        (2) The land cover map results
        (3) The first-round interpretation results
        (4) The exclude flag
    """

    reverse_validation_system = {v: int(k) for k, v in landcover_system_merge.items()}

    array_count_merge_barren_cropland, array_weight_merge_barren_cropland = get_count_weight_merge_barren_cropland(landcover_version='degrade_v2_refine_3_3')

    path_validation = join(rootpath, 'results', 'accuracy_assessment', 'degrade_v2_refine_3_3_separate_validation', 'lc_final_sample')

    # read the spreadsheet including the land cover classification results
    filename_lc_map = join(path_validation, f'validation_lc_400_python_output_fixed_for_share.xlsx')
    sheet_lc_map = pd.read_excel(filename_lc_map, sheet_name='Sheet1')

    array_lc_map = sheet_lc_map['map_type'].values
    array_lc_map = np.array([reverse_validation_system.get(i, -999) for i in array_lc_map])

    path_validation = join(rootpath, 'results', 'accuracy_assessment', 'degrade_v2_refine_3_3_separate_validation', 'lc_final_sample')

    filename_validation = join(path_validation, 'collection_v2_validation_lc_400samples.xlsx')
    df_interpretation = pd.read_excel(filename_validation, sheet_name='validation_record')

    array_lc_validation = df_interpretation['pri_lc'].values
    array_lc_validation = np.array([reverse_validation_system.get(i, -999) for i in array_lc_validation])

    array_exclude = df_interpretation['exclude_flag'].values == True

    return array_lc_map, array_weight_merge_barren_cropland, array_count_merge_barren_cropland, array_lc_validation, array_exclude, reverse_validation_system


# def main():
if __name__ == '__main__':

    landcover_system_merge = {'1': 'developed',
                              '2': 'barren/cropland',
                              '3': 'primary wet forest',
                              '4': 'primary dry forest',
                              '5': 'secondary forest',
                              '6': 'shrub/grass',
                              '7': 'water',
                              '8': 'wetland'}

    pf_loss_dict = {'1': 'Non PF loss',
                    '2': 'PF loss'}

    (array_pf_loss_map, array_weight_pf_loss, array_counts_pf_loss,
     array_pf_loss_first_round_validation, array_pf_loss_first_round_exclude,
     reverse_pf_loss_system) = prepare_pf_loss_data()

    (array_lc_map, array_weight_lc, array_counts_lc,
     array_lc_first_round_validation, array_lc_first_round_exclude,
     reverse_lc_system) = prepare_lc_map_data()

    ##

    filename_collection = join(rootpath, 'results', 'accuracy_assessment', 'degrade_v2_refine_3_3_separate_validation',
                               're_interpretation_sample',
                               'Collection_re_interpretation_sample.xlsx')

    sheet_lc_validation = pd.read_excel(filename_collection, sheet_name='land_cover_reinterpretation')
    index_lc = sheet_lc_validation['Index'].values
    exclude_flag_lc = sheet_lc_validation['final_exclude_flag'].values == True

    array_lc = sheet_lc_validation['final_pri_lc'].values
    array_lc = np.array([reverse_lc_system.get(i, -999) for i in array_lc])

    sheet_pf_loss_validation = pd.read_excel(filename_collection, sheet_name='pf_loss_reinterpretation')
    index_pf_loss = sheet_pf_loss_validation['Index'].values
    exclude_flag_pf_loss = sheet_pf_loss_validation['final_exclude_flag'].values == True

    array_pf_loss = sheet_pf_loss_validation['final_val_label'].values
    array_pf_loss = np.array([reverse_pf_loss_system.get(i, -999) for i in array_pf_loss])

    ##

    array_lc_all_new = array_lc_first_round_validation.copy()
    array_lc_all_new[index_lc - 1] = array_lc

    mask_exclude_lc = np.ones(len(array_lc_first_round_validation), dtype=bool)
    mask_exclude_lc[(array_lc_map == -999) | (array_lc_all_new == -999) | (array_lc_all_new == 999)] = False
    mask_exclude_lc[array_lc_first_round_exclude] = False

    df_confusion_lc, df_err_adjust_lc = summary_accuracy_evaluation(array_lc_map, array_lc_all_new, mask_exclude_lc,
                                                                    array_weight_lc, array_counts_lc)
    plot_df_confusion(df_confusion_lc.values, landcover_system=landcover_system_merge, title='final-round-reinterpretation', figsize=(11.5, 8))

    ##

    # array_lc_reinterpretation_flag = np.zeros(len(array_lc_map), dtype=bool)
    # array_lc_reinterpretation_flag[index_lc - 1] = True
    #
    # for p in array_lc_reinterpretation_flag:
    #     print(p)

    ##

    array_pf_loss_all_new = array_pf_loss_first_round_validation.copy()
    array_pf_loss_all_new[index_pf_loss - 1] = array_pf_loss

    mask_exclude = np.ones(len(array_pf_loss_first_round_validation), dtype=bool)
    mask_exclude[(array_pf_loss_first_round_validation == -999) | (array_pf_loss_all_new == -999) | (array_pf_loss_all_new == 999)] = False
    mask_exclude[array_pf_loss_first_round_exclude] = False

    df_confusion_pf_loss, df_err_adjust_pf_loss, f1_score_pf_loss = validation_pf_loss_summary(array_pf_loss_map[mask_exclude], array_pf_loss_all_new[mask_exclude],
                                                                               array_weight_pf_loss, array_counts_pf_loss)
    plot_df_confusion(df_confusion_pf_loss.values, landcover_system=pf_loss_dict, title='final-round-reinterpretation', figsize=(11.5, 8))

    ##
    # array_pf_loss_reinterpretation_flag = np.zeros(len(array_pf_loss_map), dtype=bool)
    # array_pf_loss_reinterpretation_flag[index_pf_loss - 1] = True
    #
    # for p in array_pf_loss_reinterpretation_flag:
    #     print(p)