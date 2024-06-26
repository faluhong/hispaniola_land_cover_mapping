#!/usr/bin/env python
# coding: utf-8

"""
COLO running
input parameters:
(3) output_foldername: the output folder name, which is related to the COLD setting.
    output_foldername == 'COLD_output': the basic COLD setting, conse = 6, change_probability = chi2.ppf(0.99,5) / 15.0863
    output_foldername == 'COLD_output_morechange': the COLD setting with more change, conse=4, change_probability = chi2ppf(0.95,5) / 11.07
    output_foldername == 'COLD_output_morechange_lessgap': the COLD setting with more change, conse=4,
    change_probability = chi2ppf(0.95,5) / 11.07, gap_days = 1500

"""


import time
import numpy as np
import os
from os.path import join
import sys
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import click
from pycold import cold_detect
from osgeo import gdal, gdal_array
import logging

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)
# print('pwd:', pwd)
# print('rootpath project:', rootpath)
# print('path code:', path_pythoncode)

# from Basic_tools.Figure_plot import FP_ISP, FP, band_plot, RGB_composite, qc_plot
from Basic_tools.datetime_datenum_convert import datenum_to_datetime, datetime_to_datenum
from Basic_tools.standard_CCD import standard_CCD

from COLD.cold_running_utils import (filename_sort, stacking_image_acquire, redundancy_filename_remove, remove_lt_tier2,
    obs_datenumber_calculate, stacking_data_prepare, 
        get_task_block, get_finished_block, get_all_block, cold_output)

NBands, NRow, NCol = 8, 2500, 2500
Block_Size = 250

# rootpath_scratch = r'/scratch/zhz18039/fah20002/LCM_diversity'
# rootpath_cn451 = r'/shared/cn451/Falu/LCM_diversity'


def cold_data_prepare(list_filename):
    """
    prepare the data for COLD running, major steps include:
    (1) sort the filename based on the chronological order, from ord to new, using the 'filename_sort' function
    (2) remove the redundant file in the same observation date coming from different rows and same path,
    using 'redundancy_filename_remove' function
    (3) prepare the data for COLD running, which includes:
        (i) separate the observations into different path;
        (ii) get the observation number
        (iii) file name list
        (iv) stacking image

        using the 'stacking_data_prepare' function

    Args :
        list_filename: the input list filename
    Returns:
        the dataframe containing: (1) the path id; (2) the list of file name within the corresponding path;
        (3) the corresponding observation date number (4) the stacking image
    """

    list_filename = filename_sort(list_filename)

    list_filename = redundancy_filename_remove(list_filename)
    
    list_filename = remove_lt_tier2(list_filename)

    df2 = stacking_data_prepare(list_filename)

    return df2


def cold_running(df_cold, blockname, img_wrsid, output_foldername, running_direction):
    """
        running the COLD for each block

        Args:
            df_cold: the dataframe including the path id, the observation number array, the stacking image
            blockname: the blockname for running
            img_wrsid: the WRS file including the singlepath id
            output_foldername: COLD output foldername, which determines the COLD parameter setting
        Returns:
            COLD output rec_cg
    """

    row_id_block = int(blockname[3:7])
    col_id_block = int(blockname[10:14])

    cold_result = []
    for row_id in range(0, Block_Size):
        for col_id in range(0, Block_Size):

            row_id_intile = row_id_block + row_id
            col_id_intile = col_id_block + col_id

            singlepathid = '%03d' % (img_wrsid[row_id_intile, col_id_intile])

            if ((df_cold['pathid'] == singlepathid).values).any() == False:
                # print(row_id, col_id, 'No corresponding Landsat observations')
                continue

            img_stacking = df_cold['stacking_data'][(df_cold['pathid'] == singlepathid).values].values[0]
            array_obsdatenum = df_cold['obsnum'][(df_cold['pathid'] == singlepathid).values].values[0]

            blues = img_stacking[:, 0, row_id, col_id].astype(np.int64)
            greens = img_stacking[:, 1, row_id, col_id].astype(np.int64)
            reds = img_stacking[:, 2, row_id, col_id].astype(np.int64)
            nirs = img_stacking[:, 3, row_id, col_id].astype(np.int64)
            swir1s = img_stacking[:, 4, row_id, col_id].astype(np.int64)
            swir2s = img_stacking[:, 5, row_id, col_id].astype(np.int64)
            thermals = img_stacking[:, 6, row_id, col_id].astype(np.int64)
            qas = img_stacking[:, 7, row_id, col_id].astype(np.int64)
            
            if running_direction == 'forward':
                pass
            elif running_direction == 'backward':
                array_obsdatenum = -array_obsdatenum[::-1]
                blues = blues[::-1]
                greens = greens[::-1]
                reds = reds[::-1]
                nirs = nirs[::-1]
                swir1s = swir1s[::-1]
                swir2s = swir2s[::-1]
                thermals = thermals[::-1]
                qas = qas[::-1]
            elif running_direction == 'backward_adjust':
                
                list_band_name = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Thermal']
                
                obs_data = np.array([array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas])
                df_example = pd.DataFrame(obs_data.T, columns=['dates'] + list_band_name + ['qa'])
                
                # adjust the observation dates
                target_obs_datenum = datetime_to_datenum(datetime(year=1992, month=1, day=1))
                array_obsdatenum[array_obsdatenum < target_obs_datenum] = array_obsdatenum[array_obsdatenum < target_obs_datenum] + 365.25 * 4
                
                # sort the observation dates from descending order
                df_example['dates'] = array_obsdatenum
                df_example = df_example.sort_values(by='dates')
                
                array_obsdatenum = df_example['dates'].values
                blues =  df_example['Blue'].values
                greens =  df_example['Green'].values
                reds =  df_example['Red'].values
                nirs =  df_example['NIR'].values
                swir1s =  df_example['SWIR1'].values
                swir2s =  df_example['SWIR2'].values
                thermals =  df_example['Thermal'].values
                qas =  df_example['qa'].values
                
                # prepare the dataset for backward COLD running
                array_obsdatenum = -array_obsdatenum[::-1]
                blues = blues[::-1]
                greens = greens[::-1]
                reds = reds[::-1]
                nirs = nirs[::-1]
                swir1s = swir1s[::-1]
                swir2s = swir2s[::-1]
                thermals = thermals[::-1]
                qas = qas[::-1]
            
            pos = row_id * Block_Size + col_id + 1
            # the thermal band will change after cold running
            # thermals = thermals *10 -27320

            try:
                if output_foldername == 'COLD_output':
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, b_c2=True, gap_days=1500)
                elif output_foldername == 'COLD_output_morechange':
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, conse=4, t_cg=11.07, b_c2=True)
                elif output_foldername == 'COLD_output_morechange_gap1500':
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, conse=4, t_cg=11.07, b_c2=True,
                                                          gap_days=1500)
                elif output_foldername == 'COLD_output_morechange_gap365':
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, conse=4, t_cg=11.07, b_c2=True,
                                                          gap_days=365)
                elif output_foldername == 'backward_COLD_output_morechange_gap1500':
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, conse=4, t_cg=11.07, b_c2=True,
                                                          gap_days=1500)
                else:
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, conse=4, t_cg=11.07, b_c2=True,
                                                          gap_days=1500)

                cold_result.append(cold_result_singlepixel)
            except Exception as e:
                pass
                # print(blockname, 'COLD failed',e)

    return cold_result


@click.command()
@click.option('--rank', type=int, default=0, help='job array id, e.g., 0-199')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
@click.option('--output_foldername', type=str, default='COLD_output')
@click.option('--rootpath_scratch', type=str, help='rootpath of the project folder', default='/scratch/zhz18039/fah20002/LCM_diversity')
@click.option('--running_direction', type=str, help='COLD is ran forward or backward', default='forward')
def main(rank, n_cores, output_foldername, rootpath_scratch, running_direction):

# if __name__ == "__main__":

#     rank, n_cores = 62, 500
#     output_foldername = 'forward_extension_v4'
#     rootpath_scratch = r'/scratch/zhz18039/fah20002/LCM_diversity'
#     running_direction = 'backward'

    starttime = time.perf_counter()

    output_rootfolder = join(rootpath_scratch, 'results', output_foldername)
    if not os.path.exists(output_rootfolder):
        os.makedirs(output_rootfolder, exist_ok=True)

    logging.basicConfig(filename=join(output_rootfolder, 'cold_running.log'),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.info('cold output folder name is {}'.format(output_foldername))
    logging.info('cold running direction {}'.format(running_direction))

    output_txt = join(output_rootfolder, 'COLD_running_finished.txt')

    iteration_flag = 0
    while os.path.exists(output_txt) != True:

        logging.info('rank: {}, iteration {}'.format(rank, iteration_flag))

        df_all_block = get_all_block(rootpath_scratch)
        df_finish_block = get_finished_block(output_rootfolder)
        df_task = get_task_block(df_all_block, df_finish_block)

        count_block = len(df_task)
        logging.info('rank: {}, block number: {}'.format(rank, count_block))

        if (count_block == 0):
            print('all blocks have finished')
            f = open(output_txt, 'w')
            f.write('COLD running all blocks finished')
            f.close()
            logging.info('{} all cold running finished'.format(rank))

        elif (count_block != 0) & (iteration_flag > 20):
            print('maximum iteration times reach')
            f = open(output_txt, 'w')
            f.write('maximum iteration times reach, count of task block {}'.format(len(df_task)))
            f.close()
            logging.info('{} maximum iteration times reach'.format(rank))

        else:
            each_core_block = int(np.ceil(count_block / n_cores))
            for i in range(0, each_core_block):

                new_rank = rank - 1 + i * n_cores
                print('running rank:{}'.format(new_rank))

                if new_rank > count_block - 1:  # means that all folder has been processed
                    print('this is the last running task')
                else:
                    tilename = df_task.loc[new_rank, 'tilename']
                    blockname = df_task.loc[new_rank, 'blockname']

                    img_wrsid = gdal_array.LoadFile(join(rootpath_scratch, 'data', 'shapefile', 'singlepath', '{}.tif'.format(tilename)))
                    # FP(img_wrsid, title='WRS id')

                    print('tilename: {}, blockname:{}'.format(tilename, blockname))
                    logging.info('rank {}: COLD running start: tilename: {}, blockname:{}'.format(rank, tilename, blockname))

                    block_path = join(rootpath_scratch, 'data', 'ARD_block', tilename, blockname)
                    list_filename = glob.glob(join(block_path, '*.npy'))

                    df_cold = cold_data_prepare(list_filename)
                    cold_result = cold_running(df_cold, blockname, img_wrsid, output_foldername, running_direction)
                    cold_output(cold_result, output_rootfolder, tilename, blockname)

                    logging.info('rank {}: COLD running end: tilename: {}, blockname:{}'.format(rank, tilename, blockname))

        iteration_flag += 1

    endtime = time.perf_counter()
    print(rank, 'running time is', endtime - starttime)


if __name__ == "__main__":
    main()
