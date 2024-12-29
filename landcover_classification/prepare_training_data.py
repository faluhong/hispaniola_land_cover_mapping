"""
    example of preparing the training data

    The final training data is in: data/rf_training/x_training_refine_i0.npy and data/rf_training/y_training_refine_i0.npy
"""

import numpy as np
import os
from os.path import join
import pandas as pd
from datetime import datetime

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))

from landcover_classification.lc_classification import read_dem, prepare_xdata_with_topography


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

        y_data_each_block, list_cold_training = training_sample_preparation_cold_reccg(df_eachblock)
        if len(y_data_each_block) > 0:
            array_cold_training_each_block = np.array([list_cold_training]).T[:, 0]

            x_data_each_block = prepare_xdata_with_topography(array_cold_training_each_block,
                                                             img_dem_block, img_slope_block, img_aspect_block)

            x_training_withdem.append(x_data_each_block)
            y_training_withdem.append(y_data_each_block)

    x_training_withdem = np.concatenate(x_training_withdem, axis=0)
    y_training_withdem = np.concatenate(y_training_withdem, axis=0)

    return x_training_withdem, y_training_withdem


def training_sample_preparation_cold_reccg(df_interpretation_sample):
    """
        get the training sample and list of COLD reccg (prepare for getting the training data)
        Args:
            df_interpretation_sample:
        Returns:
            y_training: the y training data
            list_cold_reccg_training: list of the COLD reccg for the matched training samples
    """

    list_blockname = df_interpretation_sample['tilename'].values + df_interpretation_sample['blockname'].values
    list_unique = list(set(list_blockname))

    list_cold_reccg_training = []  # list to store the COLD rec_cg for the training samples
    y_training = np.array([], dtype=float)

    mask_match_df = np.zeros(len(df_interpretation_sample), dtype=bool)

    for i_unique, unique in enumerate(list_unique):
        tilename, blockname = unique[0:6], unique[6:20]
        print(tilename, blockname)

        filename_cold_reccg = join(rootpath, 'data', 'cold_reccg', tilename, '{}_reccg.npy'.format(blockname))
        if os.path.exists(filename_cold_reccg) == False:
            print('{} not exist'.format(filename_cold_reccg))
            pass
        else:
            # print(i_unique, tilename, blockname)
            cold_reccg = np.load(filename_cold_reccg)  # read the COLD outputs

            mask_df = (df_interpretation_sample['tilename'] == tilename) & (df_interpretation_sample['blockname'] == blockname)
            df_interpretation_sample_each_block = df_interpretation_sample[mask_df].copy()

            list_cold_reccg_training_each_block = []   # list to store the COLD rec_cg for the training samples
            mask_training = np.ones(len(df_interpretation_sample_each_block), dtype=bool)   # mask to identify whether there is a match for the training sample

            for i_row in range(0, len(df_interpretation_sample_each_block)):

                row_id_in_block = df_interpretation_sample_each_block.loc[df_interpretation_sample_each_block.index[i_row], 'row_id_in_block']
                col_id_in_block = df_interpretation_sample_each_block.loc[df_interpretation_sample_each_block.index[i_row], 'col_id_in_block']
                Block_size = 250

                pos_trainingsample = row_id_in_block * Block_size + col_id_in_block + 1  # the pos parameter in the COLD_reccg

                year = df_interpretation_sample_each_block.loc[df_interpretation_sample_each_block.index[i_row], 'year']
                target_obsnum = datetime_to_datenum(datetime(year=year, month=7, day=1))

                cold_results_trainingsample = cold_reccg[cold_reccg['pos'] == pos_trainingsample]  # get the COLD results for the training sample

                if len(cold_results_trainingsample) == 0:
                    print('{}: 1 no matching COLD results at tile {} in row:{} col:{}'.format(i_row, tilename, row_id_in_block, col_id_in_block))
                    mask_training[i_row] = False
                else:
                    match_flag = 0
                    for i in range(0, len(cold_results_trainingsample)):
                        t_start = cold_results_trainingsample[i]['t_start']
                        t_end = cold_results_trainingsample[i]['t_end']

                        if (t_start <= target_obsnum) & (t_end >= target_obsnum):
                            match_flag = 1
                            # print(i_row, 'matched type 1', len(list_COLD_training_eachblock))
                            break

                        # if the training sample interpretation date is 2022 while the COLD rec_cg does not reach the data,
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
                        # There is a match between the training sample and COLD records, so append it to the output
                        list_cold_reccg_training_each_block.append(cold_results_trainingsample[i])

            y_training_each_block = df_interpretation_sample_each_block['landcover_type'][mask_training].values

            mask_match_df[mask_df] = mask_training

            # append the COLD reccg from each block
            for p in list_cold_reccg_training_each_block:
                list_cold_reccg_training.append(p)

            # append the y_training from each block
            y_training = np.concatenate([y_training, y_training_each_block])
            y_training = y_training.astype(float)

    # df_training_sample.loc[:, 'match_flag'] = mask_match_df

    return y_training, list_cold_reccg_training


def datetime_to_datenum(dt):
    python_datenum = dt.toordinal()
    return python_datenum


if __name__ == "__main__":

    # read the example spreadsheet of the training sample
    filename_interpretation_sample = join(rootpath, 'data', 'rf_training', 'example_training_sample.xlsx')
    sheet_interpretation_sample = pd.read_excel(filename_interpretation_sample, sheet_name='Sheet1')

    x_training_with_topography, y_training_with_topography = get_x_y_training_withdem(sheet_interpretation_sample)
    print('number of example training sample is {}'.format(len(y_training_with_topography)))

    # random forest training. Training 25000 sample takes about 2 minutes
    from landcover_classification.rf_training import rf_training
    rf_classifier = rf_training(x_training_with_topography, y_training_with_topography)




