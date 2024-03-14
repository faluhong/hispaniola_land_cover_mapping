"""
    prepare the training data from COLD reccg
"""

import time
import joblib
import numpy as np
import os
from os.path import join
import sys
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import json as js
import click
from osgeo import gdal, gdal_array, gdalconst
import sklearn
from sklearn.ensemble import RandomForestClassifier
import heapq
import click
import fiona
import logging
import warnings
warnings.filterwarnings("ignore")

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

from landcover_classification import read_dem, prepare_xdata_with_topography

def rf_training(x_trainingdata, y_trainingdata):
    """
    training the random forest model
    Args:
        x_trainingdata
        y_trainingdata
    Returns:
        RF_classifier
    """

    starttime_training = time.perf_counter()
    rf_classifier = RandomForestClassifier(n_estimators=500, random_state=0)  # fix the random_state parameter to
    rf_classifier.fit(x_trainingdata, y_trainingdata)
    end_time_training = time.perf_counter()
    print('random forest training time:', end_time_training - starttime_training)
    return rf_classifier


def datetime_to_datenum(dt):
    python_datenum = dt.toordinal()
    return python_datenum


def y_label_and_cold_segment_extraction(sheet_interpretation_sample_each_block):
    """
        get the y training label and the corresponding matched COLD reccg segment for each block
        Not all the interpretation samples have the matched COLD reccg segment.
        Because: (1) July 1st is between the COLD segment, i.e., during change
                 (2) The COLD time series is not long enough to cover the target date, for example, the sample was interpreted in 2023, but the COLD time series only cover 2022

        Args:
            sheet_interpretation_sample_each_block: The dataframe containing the interpretation sample information for each block
        Returns:
            y_training: the y training data
            list_COLD_training: list of the COLD reccg
    """

    list_blockname = sheet_interpretation_sample_each_block['tilename'].values + sheet_interpretation_sample_each_block['blockname'].values
    list_unique = list(set(list_blockname))
    tilename, blockname = list_unique[0][0:6], list_unique[0][6:20]   # get the tile name and block name. Usually contains only one block

    list_cold_reccg_training = []   # the list to store the COLD segments which match the training sample interpretation
    y_training = np.array([], dtype=float)   # the y training data, i.e., the labelled land cover

    filename_cold_reccg = join(rootpath, 'data', 'cold_reccg', tilename, '{}_reccg.npy'.format(blockname))
    if os.path.exists(filename_cold_reccg) == False:
        print('{} not exist'.format(filename_cold_reccg))
        pass
    else:
        cold_reccg = np.load(filename_cold_reccg)

        mask_df = (sheet_interpretation_sample_each_block['tilename'] == tilename) & (sheet_interpretation_sample_each_block['blockname'] == blockname)
        df_eachblock = sheet_interpretation_sample_each_block[mask_df].copy()

        list_cold_training_eachblock = []
        mask_training = np.ones(len(df_eachblock), dtype=bool)

        for i_row in range(0, len(df_eachblock)):

            row_id_in_block = df_eachblock.loc[df_eachblock.index[i_row], 'row_id_in_block']
            col_id_in_block = df_eachblock.loc[df_eachblock.index[i_row], 'col_id_in_block']
            Block_size = 250

            pos_training_sample = row_id_in_block * Block_size + col_id_in_block + 1  # the pos parameter in the COLD_reccg, starting from 1

            # get the year of the training sample
            # we use the July 1st at the interpretation year as the target date to match the COLD segment
            year = df_eachblock.loc[df_eachblock.index[i_row], 'year']
            target_obsnum = datetime_to_datenum(datetime(year=year, month=7, day=1))

            cold_results_trainingsample = cold_reccg[cold_reccg['pos'] == pos_training_sample]   # get the COLD reccg for the selected training sample

            if len(cold_results_trainingsample) == 0:
                print('No COLD fitting at tile {} in row:{} col:{}'.format(tilename, row_id_in_block, col_id_in_block))
                mask_training[i_row] = False
            else:
                match_flag = 0
                for i in range(0, len(cold_results_trainingsample)):
                    t_start = cold_results_trainingsample[i]['t_start']
                    t_end = cold_results_trainingsample[i]['t_end']

                    if (t_start <= target_obsnum) & (t_end >= target_obsnum):
                        # find the matched COLD segment
                        match_flag = 1
                        break

                if match_flag == 0:
                    # There is no matched COLD segment, because:
                    # (1) July 1st is between the COLD segmement, i.e., during change
                    # (2) The COLD time series is not long enough to cover the target date, for example, the sample was interpreted in 2023, but the COLD time series only cover 2022
                    print('No matched COLD segment at tile {} in row:{} col:{}'.format(tilename, row_id_in_block, col_id_in_block))
                    mask_training[i_row] = False
                else:
                    list_cold_training_eachblock.append(cold_results_trainingsample[i])  # store the matched COLD segment

        y_training_eachblock = df_eachblock['landcover_type'][mask_training].values

        list_cold_reccg_training.extend(list_cold_training_eachblock)

        y_training = np.concatenate([y_training, y_training_eachblock])
        y_training = y_training.astype(float)

    return y_training, list_cold_reccg_training


def get_x_y_training_data(sheet_interpretation_sample):
    """
        get the
        Args:
            df_selectedsamples: the dataframe containing the information of selected training samples
        Returns:
            x_training_withdem, y_training_withdem
    """

    # get the unique ID of each block in the interpretation sample.
    # This is for getting the training data for the whole block by reading the COLD reccg file only once to save the computation time.
    list_blockname = sheet_interpretation_sample['tilename'].values + sheet_interpretation_sample['blockname'].values
    list_unique = list(set(list_blockname))

    x_training_with_topography = []
    y_training_with_topography = []

    for i_unique, unique in enumerate(list_unique):
        tilename, blockname = unique[0:6], unique[6:20]

        # read the topography information. If you do not need the topography information, you can remove that accordingly
        img_dem_block, img_slope_block, img_aspect_block = read_dem(tilename, blockname)

        # get the dataframe which contains the collected training samples in the block
        mask_df = (sheet_interpretation_sample['tilename'] == tilename) & (sheet_interpretation_sample['blockname'] == blockname)
        df_eachblock = sheet_interpretation_sample[mask_df].copy()

        # get y-training label and list of corresponding COLD reccg segment
        y_training_label_each_block, list_cold_training = y_label_and_cold_segment_extraction(df_eachblock)
        array_cold_training_each_block = np.array([list_cold_training]).T[:, 0]   # conver the list to array for the next step

        # prepare the x-training data containing the topography information
        x_data_eachblock = prepare_xdata_with_topography(array_cold_training_each_block, img_dem_block, img_slope_block, img_aspect_block)

        x_training_with_topography.append(x_data_eachblock)
        y_training_with_topography.append(y_training_label_each_block)

    x_training_with_topography = np.concatenate(x_training_with_topography, axis=0)
    y_training_with_topography = np.concatenate(y_training_with_topography, axis=0)

    return x_training_with_topography, y_training_with_topography


# def main():
if __name__ == "__main__":

    # read the interpretation sample spreadsheet
    # The spreadsheet is a fake data used for illustration and testing.
    # You need to prepare this by yourself based on how you collect the training samples.
    filename_interpretation_sample = join(rootpath, 'data', 'rf_training', 'example_training_sample.xlsx')
    sheet_interpretation_sample = pd.read_excel(filename_interpretation_sample, sheet_name='Sheet1')

    # get the training data
    x_training_withdem, y_training_withdem = get_x_y_training_data(sheet_interpretation_sample)
    print('number of training sample is {}'.format(len(y_training_withdem)))

    # random forest training. Training 25000 sample takes about 2 minutes
    rf_classifier = rf_training(x_training_withdem, y_training_withdem)

    # save the random forest model
    output_folder_rf_classifier = join(rootpath, 'results', 'rf_output')
    if not os.path.exists(output_folder_rf_classifier):
        os.makedirs(output_folder_rf_classifier, exist_ok=True)

    output_filename_rf_classifier = join(output_folder_rf_classifier, 'rf_classifier_test.joblib')
    joblib.dump(rf_classifier, output_filename_rf_classifier)



