"""
    processing of running the COLD algorithm parallelly in the HPC environment
"""

import numpy as np
import os
from os.path import join
import glob
import click
import pandas as pd
from pycold import cold_detect  # pycold is workable on Mac and Linux system. Windows system is not workable at present
from osgeo import gdal, gdal_array
import logging

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))

from cold_running.cold_running import (cold_data_prepare, cold_running, cold_output)

NBands, NRow, NCol = 8, 2500, 2500
Block_Size = 250


def get_all_block(rootpath_scratch):
    """
        get all the stored blocks

        Args:
        Returns:
            The dataframe containing the tile name and block name for all blocks
    """

    list_tilename = ['h00v00', 'h00v01', 'h00v02', 'h00v03',
                     'h01v00', 'h01v01', 'h01v02', 'h01v03',
                     'h02v00', 'h02v01', 'h02v02', 'h02v03',
                     'h03v00', 'h03v01', 'h03v02', 'h03v03',
                     'h04v00', 'h04v01', 'h04v02', 'h04v03',
                     'h05v00', 'h05v01', 'h05v02', 'h05v03',
                     'h06v00', 'h06v01', 'h06v02', 'h06v03',
                     'h07v00', 'h07v01', 'h07v02',
                     'h08v00', 'h08v01', 'h08v02']

    df_total_block = pd.DataFrame(columns=['rank', 'tilename', 'blockname'])

    rank_id = 0
    for i_tilename, tilename in enumerate(list_tilename):
        list_block = os.listdir(join(rootpath_scratch, 'data', 'ARD_block', tilename))
        for i_block, block_id in enumerate(list_block):
            df_total_block.loc[rank_id, 'rank'] = rank_id
            df_total_block.loc[rank_id, 'tilename'] = tilename
            df_total_block.loc[rank_id, 'blockname'] = block_id
            rank_id += 1

    print('number of total block is {}'.format(len(df_total_block)))

    return df_total_block


def get_finished_block(output_rootfolder):
    """
        get all the finished blocks

        Args:
        Returns:
            The dataframe containing the tile name and block name for all finished blocks
    """

    df_finished_block = pd.DataFrame(columns=['rank', 'tilename', 'blockname'])
    list_finished_tile = os.listdir(output_rootfolder)

    rank_id = 0
    for i_tilename, tilename in enumerate(list_finished_tile):

        list_finished_reccg = glob.glob(join(output_rootfolder, tilename, '*.npy'))

        for i_reccg, reccg_filename in enumerate(list_finished_reccg):
            block_id = os.path.split(reccg_filename)[-1][0:14]

            df_finished_block.loc[rank_id, 'rank'] = rank_id
            df_finished_block.loc[rank_id, 'tilename'] = tilename
            df_finished_block.loc[rank_id, 'blockname'] = block_id
            rank_id += 1

    print('number of finished block is {}'.format(len(df_finished_block)))

    return df_finished_block


def get_task_block(df_total_block, df_finished_block):
    """
        get the block info for running, i.e., exclude the finished blocks from all blocks

        Args:
            df_total_block: dataframe containing the information of all blocks
            df_finished_block: dataframe containing the information of finished blocks
        Returns:
            The dataframe containing the tile name and block name for task blocks (waiting for running)
    """

    df_running_block = pd.DataFrame(columns=['rank', 'tilename', 'blockname'])

    rank_id = 0
    for i_rank in range(0, len(df_total_block)):
        tilename = df_total_block.loc[i_rank, 'tilename']
        blockname = df_total_block.loc[i_rank, 'blockname']

        mask_tilename = df_finished_block['tilename'].values == tilename
        mask_blockname = df_finished_block['blockname'].values == blockname

        if np.count_nonzero(mask_tilename & mask_blockname) == 1:
            pass
            # print('COLD at {} {} has finished'.format(tilename, blockname))
        else:
            df_running_block.loc[rank_id, 'rank'] = rank_id
            df_running_block.loc[rank_id, 'tilename'] = tilename
            df_running_block.loc[rank_id, 'blockname'] = blockname
            rank_id += 1

    print('number of task block is {}'.format(len(df_running_block)))

    return df_running_block

@click.command()
@click.option('--rank', type=int, default=0, help='job array id, e.g., 0-199')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
@click.option('--output_foldername', type=str, default='COLD_output', help='the folder to output the COLD results')
@click.option('--rootpath_scratch', type=str, help='rootpath of the project folder to store all data')
def main(rank, n_cores, output_foldername, rootpath_scratch):

    output_rootfolder = join(rootpath_scratch, 'results', output_foldername)
    if not os.path.exists(output_rootfolder):
        os.makedirs(output_rootfolder, exist_ok=True)

    logging.basicConfig(filename=join(output_rootfolder, 'cold_running.log'),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.info('cold output folder name is {}'.format(output_foldername))

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
                    cold_result = cold_running(df_cold, blockname, img_wrsid, output_foldername)
                    cold_output(cold_result, output_rootfolder, tilename, blockname)

                    logging.info('rank {}: COLD running end: tilename: {}, blockname:{}'.format(rank, tilename, blockname))

        iteration_flag += 1


if __name__ == "__main__":
    main()
