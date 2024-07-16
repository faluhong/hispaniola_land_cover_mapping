"""
    example of running pycold for one Block (250 * 250) size
    Major steps include:
    1. preprare the dateset --> "cold_data_prepare" function
        1.1 sort the filename
        1.2 remove the redundant files
        1.3 read the dataset
    2. running the pycold --> "cold_running" function
    3. output the COLD running results 
    
    running time is about 4 minite 20 seconds for one Block 
"""

import time
import numpy as np
import os
from os.path import join
import sys
import glob
import pandas as pd
from datetime import datetime
from pycold import cold_detect  # pycold is workable on Mac and Linux system. Windows system is not workable at present
from osgeo import gdal, gdal_array
import logging

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))

NBands, NRow, NCol = 8, 2500, 2500
Block_Size = 250


def datetime_to_datenum(dt):
    """convert the datetime to datenum

    Args:
        dt: input datetime
    Returns:
        datenum
    """
    python_datenum = dt.toordinal()
    return python_datenum


def filename_sort(list_filename):
    """
    sort the filename following the observation time order
    Args:
        list_filename: the input list filename
    Returns:
          sorted filename
    """

    series_order = pd.Series(dtype=pd.Int16Dtype())
    for i_filename in range(0, len(list_filename)):
        obs_date = os.path.split(list_filename[i_filename])[-1][17:25]

        obs_year = int(obs_date[0:4])
        obs_month = int(obs_date[4:6])
        obs_day = int(obs_date[6:8])

        obs_time = datetime(year=obs_year, month=obs_month, day=obs_day)
        
        series_order = pd.concat([series_order, pd.Series(data=i_filename, index=np.array([datetime_to_datenum(obs_time)], dtype=np.int64))])

    series_order = series_order.sort_index()

    list_filename_return = []
    for i_order, order in enumerate(series_order):
        list_filename_return.append(list_filename[order])

    return list_filename_return


def redundancy_filename_remove(list_filename):
    """
    remove the redundant observations, including two steps:
    (1) remove the image on the same observation dates, but different processing dates, for example, (a) and (c), (b) and (d):
    
    (a) LC09_L2SP_010047_20221113_20230322_02_T1
    (b) LC09_L2SP_010046_20221113_20221115_02_T1
    (c) LC09_L2SP_010047_20221113_20221115_02_T1
    (d) LC09_L2SP_010046_20221113_20230322_02_T1
    
    The older processing images will be removed 
    
    (2) remove the images on the same path but different row for example, '008048_20210506' and '008047_20210506'
    image with less clear land observations (qa==1) is removed
    
    Args:
        list_filename: the input list filename
    Returns:
        the update listfilename with the redundant file name removed
    """

    list_filename_process_date_remove = list_filename.copy()
    for i_filename in range(0, len(list_filename)):

        filename_current = list_filename[i_filename]
        path_id_current = os.path.split(filename_current)[-1][10:13]
        row_id_current = os.path.split(filename_current)[-1][13:16]

        obs_date_current = os.path.split(filename_current)[-1][17:25]
        process_date_current = os.path.split(filename_current)[-1][26:34]

        for i_filename_compare in range(i_filename + 1, len(list_filename)):
        
            filename_compare = list_filename[i_filename_compare]
            path_id_compare = os.path.split(filename_compare)[-1][10:13]
            row_id_compare = os.path.split(filename_compare)[-1][13:16]

            obs_date_compare = os.path.split(filename_compare)[-1][17:25]
            process_date_compare = os.path.split(filename_compare)[-1][26:34]

            # when two images in the same date belong to the same path but different row, images with less clear 
            # observations will be removed
            if (path_id_current == path_id_compare) & (row_id_current == row_id_compare) & (obs_date_current == obs_date_compare):
                if int(process_date_current) >= int(process_date_compare):
                    if filename_compare in list_filename_process_date_remove:
                        list_filename_process_date_remove.remove(filename_compare)
                else:
                    if filename_current in list_filename_process_date_remove:
                        list_filename_process_date_remove.remove(filename_current)
        
    list_filename_return_final = list_filename_process_date_remove.copy()
    for i_filename in range(1, len(list_filename_process_date_remove)):

        filename_current = list_filename_process_date_remove[i_filename]
        path_id_current = os.path.split(filename_current)[-1][10:13]
        row_id_current = os.path.split(filename_current)[-1][13:16]

        obs_date_current = os.path.split(filename_current)[-1][17:25]

        filename_previous = list_filename_process_date_remove[i_filename - 1]
        path_id_previous = os.path.split(filename_previous)[-1][10:13]
        row_id_previous = os.path.split(filename_previous)[-1][13:16]

        obs_date_previous = os.path.split(filename_previous)[-1][17:25]

        # when two images in the same date belong to the same path but different row, images with less clear observations will be removed
        if (path_id_current == path_id_previous) & (row_id_current != row_id_previous):
            if obs_date_previous == obs_date_current:
                
                img_qa_previous = np.load(filename_previous)[7, :, :]
                img_qa_current = np.load(filename_current)[7, :, :]

                if np.count_nonzero(img_qa_previous == 0) < np.count_nonzero(img_qa_current == 0):
                    list_filename_return_final.remove(filename_previous)
                else:
                    list_filename_return_final.remove(filename_current)

    return list_filename_return_final


def stacking_image_acquire(list_filename):
    """
    get the stacking image
    Args:
        list_filename: the input list filename
    Returns:
        stacking image
    """

    img_stacking = np.zeros((len(list_filename), NBands, Block_Size, Block_Size), dtype=np.int16)
    for i_image in range(0, len(list_filename)):
        LT_stacking_eachday = np.load(list_filename[i_image])
        img_stacking[i_image, :, :, :] = LT_stacking_eachday

    return img_stacking


def obs_datenumber_calculate(list_filename):
    """
    return the observation date through the filename
    Args:
        list_filename: the input list filename
    Returns:
        obs datenum
    """

    series_order = pd.Series(dtype=pd.Int16Dtype())
    for i_filename in range(0, len(list_filename)):
        obs_date = os.path.split(list_filename[i_filename])[-1][17:25]

        obs_year = int(obs_date[0:4])
        obs_month = int(obs_date[4:6])
        obs_day = int(obs_date[6:8])

        obs_time = datetime(year=obs_year, month=obs_month, day=obs_day)

        series_order = pd.concat([series_order, pd.Series(data=i_filename, index=np.array([datetime_to_datenum(obs_time)], dtype=np.int64))])

    return np.array(series_order.keys())


def stacking_data_prepare(list_filename):
    """
    prepare the data for COLD running based on the preprocessed file name list
    Args:
        list_filename: the input list filename
    Returns:
        the dataframe containing: (1) the path id; (2) the list of file name within the corresponding path;
        (3) the corresponding observation date number (4) the stacking image
    """

    df = pd.DataFrame(columns=['pathid', 'filename', 'obsnum', 'stacking_data'], dtype='object')

    list_pathid = []
    for i_filename, filename in enumerate(list_filename):
        pathid = os.path.split(filename)[-1][10:13]
        list_pathid.append(pathid)

    unique_pathid = list(set(list_pathid))

    if len(unique_pathid) == 1:
        df.loc[0, 'pathid'] = unique_pathid[0]
        df.loc[0, 'filename'] = list_filename
        df.loc[0, 'obsnum'] = obs_datenumber_calculate(list_filename)
        df.loc[0, 'stacking_data'] = stacking_image_acquire(list_filename)
    else:

        df.loc[0, 'pathid'] = 'allpath'
        df.loc[0, 'filename'] = list_filename
        df.loc[0, 'obsnum'] = obs_datenumber_calculate(list_filename)
        df.loc[0, 'stacking_data'] = stacking_image_acquire(list_filename)

        for i_pathid, pathid in enumerate(unique_pathid):

            list_filename_eachpath = []

            for i_filename, filename in enumerate(list_filename):
                path_id_filename = os.path.split(filename)[-1][10:13]
                if path_id_filename == pathid:
                    list_filename_eachpath.append(filename)

            df.loc[i_pathid + 1, 'pathid'] = pathid
            df.loc[i_pathid + 1, 'filename'] = list_filename_eachpath
            df.loc[i_pathid + 1, 'obsnum'] = obs_datenumber_calculate(list_filename_eachpath)
            df.loc[i_pathid + 1, 'stacking_data'] = stacking_image_acquire(list_filename_eachpath)

    return df


def cold_data_prepare(list_filename):
    """
    prepare the data for COLD running, major steps include:
    (1) sort the filename based on the chronological order, from old to new, using the 'filename_sort' function
    (2) remove the redundant file in the same observation date coming from different rows and same path, using 'redundancy_filename_remove' function
    (3) prepare the data for COLD running, which includes:
        (i) separate the observations into different path;
        (ii) get the observation number
        (iii) file name list
        (iv) stacked image

        using the 'stacking_data_prepare' function

    Args :
        list_filename: the input list filename
    Returns:
        the dataframe containing: (1) the path id; (2) the list of file name within the corresponding path;
        (3) the corresponding observation date number (4) the stacking image
    """

    list_filename = filename_sort(list_filename)

    list_filename = redundancy_filename_remove(list_filename)

    df = stacking_data_prepare(list_filename)

    return df


def cold_running(df_cold, blockname, img_wrsid, output_foldername):
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

            pos = row_id * Block_Size + col_id + 1  # pos starts from 1

            try:
                if output_foldername == 'COLD_output':
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, b_c2=True)
                elif output_foldername == 'COLD_output_morechange':
                    cold_result_singlepixel = cold_detect(array_obsdatenum, blues, greens, reds, nirs, swir1s, swir2s,
                                                          thermals, qas, pos=pos, conse=4, t_cg=11.07, b_c2=True,
                                                         )
                
                cold_result.append(cold_result_singlepixel)
            except Exception as e:
                pass
                # print('COLD failed',e)

    return cold_result


def cold_output(cold_result, output_rootfolder, tilename, blockname):
    """
        output the COLD rec_cg file

        Args:
            cold_result: COLD rec_cg file
            output_foldername: COLD output foldername
            tilename: the tile name
            blockname: the block name for running
        Returns:
            0
    """

    if len(cold_result) == 0:
        print('NO COLD fitted results in {} {}'.format(tilename, blockname))
    else:
        outputfile = join(output_rootfolder, tilename)
        if not os.path.exists(outputfile):
            os.makedirs(outputfile, exist_ok=True)

        outputfilename = join(outputfile, blockname + '_reccg.npy')
        print(outputfilename)
        np.save(outputfilename, np.hstack(cold_result))
        print('shape of output cold results', np.shape(np.hstack(cold_result)))

    return 0


def main():
    output_foldername = 'COLD_output'

    output_rootfolder = join(rootpath, 'results', output_foldername)
    if not os.path.exists(output_rootfolder):
        os.makedirs(output_rootfolder, exist_ok=True)

    logging.basicConfig(filename=join(output_rootfolder, 'cold_running.log'),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.info('cold output folder name is {}'.format(output_foldername))

    tilename = 'h00v03'
    blockname = 'row1000col1750'

    img_wrsid = gdal_array.LoadFile(join(rootpath, 'data', 'single_path', '{}.tif'.format(tilename)))

    print('tilename: {}, blockname:{}'.format(tilename, blockname))
    logging.info('COLD running start: tilename: {}, blockname:{}'.format(tilename, blockname))

    block_path = join(rootpath, 'data', 'cold_running', tilename, blockname)
    list_filename = glob.glob(join(block_path, '*.npy'))

    df_cold = cold_data_prepare(list_filename)
    cold_result = cold_running(df_cold, blockname, img_wrsid, output_foldername)
    cold_output(cold_result, output_rootfolder, tilename, blockname)

    logging.info('COLD running end: tilename: {}, blockname:{}'.format(tilename, blockname))


if __name__ == "__main__":
    main()
