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


def get_x_y_training_withdem(df_selectedsamples):
    """
        save the refined x_training and y_training data
        Args:
            df_selectedsamples: the dataframe containing the information of selected training samples
        Returns:
            x_training_withdem, y_training_withdem
    """

    list_blockname = df_selectedsamples['tilename'].values + df_selectedsamples['blockname'].values
    list_unique = list(set(list_blockname))

    x_training_withdem = []
    y_training_withdem = []

    for i_unique, unique in enumerate(list_unique):

        tilename, blockname = unique[0:6], unique[6:20]

        img_dem_block, img_slope_block, img_aspect_block = read_dem(tilename, blockname)

        mask_df = (df_selectedsamples['tilename'] == tilename) & (df_selectedsamples['blockname'] == blockname)
        df_eachblock = df_selectedsamples[mask_df].copy()

        y_trainingdata_eachblock, list_cold_training = training_sample_preparation(df_eachblock)
        array_cold_training_eachblock = np.array([list_cold_training]).T[:, 0]

        x_data_eachblock = prepare_xdata_with_topography(array_cold_training_eachblock, img_dem_block, img_slope_block, img_aspect_block)

        x_training_withdem.append(x_data_eachblock)
        y_training_withdem.append(y_trainingdata_eachblock)

    x_training_withdem = np.concatenate(x_training_withdem, axis=0)
    y_training_withdem = np.concatenate(y_training_withdem, axis=0)

    return x_training_withdem, y_training_withdem


def training_sample_preparation(sheet_interpretation_sample):
    """
        get the training sample and list of COLD reccg (prepare for getting the training data)
        Args:
            df_training_sample:
        Returns:
            y_training: the y training data
            list_COLD_training: list of the COLD reccg
    """

    list_blockname = sheet_interpretation_sample['tilename'].values + sheet_interpretation_sample['blockname'].values
    list_unique = list(set(list_blockname))

    list_COLD_training = []
    y_training = np.array([], dtype=float)

    mask_match_df = np.zeros(len(sheet_interpretation_sample), dtype=bool)

    for i_unique, unique in enumerate(list_unique):
        tilename, blockname = unique[0:6], unique[6:20]
        print(tilename, blockname)

        filename_cold_reccg = join(rootpath, 'data', 'cold_reccg', tilename, '{}_reccg.npy'.format(blockname))
        if os.path.exists(filename_cold_reccg) == False:
            print('{} not exist'.format(filename_cold_reccg))
            pass
        else:
            # print(i_unique, tilename, blockname)
            cold_reccg = np.load(filename_cold_reccg)

            mask_df = (sheet_interpretation_sample['tilename'] == tilename) & (sheet_interpretation_sample['blockname'] == blockname)
            df_eachblock = sheet_interpretation_sample[mask_df].copy()
            print(len(df_eachblock))

            list_COLD_training_eachblock = []
            mask_training = np.ones(len(df_eachblock), dtype=bool)

            for i_row in range(0, len(df_eachblock)):

                row_id_in_block = df_eachblock.loc[df_eachblock.index[i_row], 'row_id_in_block']
                col_id_in_block = df_eachblock.loc[df_eachblock.index[i_row], 'col_id_in_block']
                Block_size = 250

                pos_trainingsample = row_id_in_block * Block_size + col_id_in_block + 1  # the pos parameter in the COLD_reccg

                year = df_eachblock.loc[df_eachblock.index[i_row], 'year']
                target_obsnum = datetime_to_datenum(datetime(year=year, month=7, day=1))

                COLD_results_trainingsample = cold_reccg[cold_reccg['pos'] == pos_trainingsample]

                if len(COLD_results_trainingsample) == 0:
                    print('{}: 1 no matching COLD results at tile {} in row:{} col:{}'.format(i_row, tilename, row_id_in_block, col_id_in_block))
                    mask_training[i_row] = False
                else:
                    match_flag = 0
                    for i in range(0, len(COLD_results_trainingsample)):
                        t_start = COLD_results_trainingsample[i]['t_start']
                        t_end = COLD_results_trainingsample[i]['t_end']

                        if (t_start <= target_obsnum) & (t_end >= target_obsnum):
                            match_flag = 1
                            # print(i_row, 'matched type 1', len(list_COLD_training_eachblock))
                            break

                        # if the training sample intrepretation date is 2022 while the COLD rec_cg does not reach the data,
                        # if the COLD rec_cg ends in 2022, then the COLD rec_cg will be select
                        elif (t_end < target_obsnum) & (t_end > datetime_to_datenum(datetime(year=2022, month=1, day=1))) \
                                & (target_obsnum == datetime_to_datenum(datetime(year=2022, month=7, day=1))):
                            # print('3 matched sample in 2022 at tile {} in row:{} col:{}'.format(tilename, row_id_in_block, col_id_in_block))
                            match_flag = 1
                            # print(i_row, 'matched type 2', len(list_COLD_training_eachblock))
                            break

                    if match_flag == 0:
                        print('{}: 2 no matching COLD results at tile {} in row:{} col:{}'.format(i_row, tilename, row_id_in_block, col_id_in_block))
                        mask_training[i_row] = False
                    else:
                        list_COLD_training_eachblock.append(COLD_results_trainingsample[i])

            y_training_eachblock = df_eachblock['landcover_type'][mask_training].values

            mask_match_df[mask_df] = mask_training

            for p in list_COLD_training_eachblock:
                list_COLD_training.append(p)
            y_training = np.concatenate([y_training, y_training_eachblock])
            y_training = y_training.astype(float)

    # df_training_sample.loc[:, 'match_flag'] = mask_match_df

    return y_training, list_COLD_training


def datetime_to_datenum(dt):
    python_datenum = dt.toordinal()
    return python_datenum


# def main():
if __name__ == "__main__":

    # read the interpretation sample
    filename_interpretation_sample = join(rootpath, 'data', 'rf_training', 'example_training_sample.xlsx')
    sheet_interpretation_sample = pd.read_excel(filename_interpretation_sample, sheet_name='Sheet1')

    ##

    get_x_y_training_withdem(sheet_interpretation_sample)
    # sheet_interpretation_sample.loc[:, 'match_flag'] = mask_match_df





