"""
The Landsat collection 2 preprocessing part

Major steps:
1. unzip the .zip file to folder 'Level2_stacking' and save he stacking file
2. clip the stacking image to each ARD block, do not save the file
3. clip each ARD file into each block, the block size is 250 by 250 pixels
"""

import numpy as np
from osgeo import gdal_array, gdal, gdalconst
import os
from os.path import join
import sys
import glob
import matplotlib.pyplot as plt
import tarfile
import click
import shutil
import fiona
import shapely
from shapely.geometry import Polygon
import time
import logging
import pandas as pd

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '../..'))
path_data = os.path.join(rootpath, 'data', 'Level2')
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)


def get_landsat_df(path_and_row):
    """
        get the list of all the Landsat filenames
        :param path_and_row

    """
    landsat_dict = {'landsat_tm_c2_l2': 0,
                    'landsat_etm_c2_l2': 0,
                    'landsat_ot_c2_l2': 0}

    df = pd.DataFrame(columns=['index', 'zip_filename', 'image_basename'])

    index = 0
    for i_landsat, landsat_collection in enumerate(list_landsat_collection):
        list_zipfile_temp = glob.glob(join(path_data, landsat_collection, path_and_row, '*.tar'))
        landsat_dict[landsat_collection] = len(list_zipfile_temp)

        for p_id, p in enumerate(list_zipfile_temp):
            image_basename = os.path.split(p)[-1][0:-4]

            df.loc[index, 'index'] = index
            df.loc[index, 'zip_filename'] = p
            df.loc[index, 'image_basename'] = image_basename

            index += 1

    df.set_index('index')

    return df


def df_task_obtain(df, path_and_row):

    df_task = df.copy()

    list_finished_filename = glob.glob(join(rootpath, 'data', 'blocking_flag', path_and_row, '*.txt'))
    for i_file, filename in enumerate(list_finished_filename):

        tmp_imagebasename = os.path.split(filename)[-1][0:-4]

        if tmp_imagebasename in df['image_basename'].values:
            index_move = df_task['index'][(df_task['image_basename'] == tmp_imagebasename)].values[0]
            df_task = df_task.drop(index=index_move)

    #         if tmp_imagebasename!=df_task.loc[index_move, 'image_basename']:
    #             print(tmp_imagebasename)
    #             print(index_move, df_task.loc[index_move, 'image_basename'])

    return df_task


def qabitval_array(packedint_array):
    """
    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value.
    fill > cloud > shadow > snow > water > clear
    Args:
        packedint: int value to bit check
    Returns:
        offset value to use
    """

    # the flag mentioned in Landsat Collection 2, reference: collection 2 level 2 guide book
    QA_DILATED_CLOUD = 1
    # QA_CIRRUS = 2
    QA_CLOUD = 3
    QA_SHADOW = 4
    QA_SNOW = 5
    QA_CLEAR = 6
    QA_WATER = 7
    QA_FILL = 255

    unpacked = np.full(packedint_array.shape, QA_FILL)

    QA_DILATED_CLOUD_unpacked = np.bitwise_and(packedint_array, 1 << QA_DILATED_CLOUD)
    # QA_CIRRUS_unpacked = np.bitwise_and(packedint_array, 1 << QA_CIRRUS)
    QA_CLOUD_unpacked = np.bitwise_and(packedint_array, 1 << QA_CLOUD)
    QA_SHADOW_unpacked = np.bitwise_and(packedint_array, 1 << QA_SHADOW)
    QA_SNOW_unpacked = np.bitwise_and(packedint_array, 1 << QA_SNOW)
    QA_CLEAR_unpacked = np.bitwise_and(packedint_array, 1 << QA_CLEAR)
    QA_WATER_unpacked = np.bitwise_and(packedint_array, 1 << QA_WATER)

    QA_CLOUD_output = 4
    QA_SHADOW_output = 2
    QA_SNOW_output = 3
    QA_CLEAR_output = 0
    QA_WATER_output = 1
    QA_FILL_output = 255

    unpacked[QA_DILATED_CLOUD_unpacked > 0] = QA_CLOUD_output
    # unpacked[QA_CIRRUS_unpacked > 0] = QA_CLOUD_output
    unpacked[QA_CLOUD_unpacked > 0] = QA_CLOUD_output

    unpacked[QA_SHADOW_unpacked > 0] = QA_SHADOW_output
    unpacked[QA_SNOW_unpacked > 0] = QA_SNOW_output
    unpacked[QA_CLEAR_unpacked > 0] = QA_CLEAR_output
    unpacked[QA_WATER_unpacked > 0] = QA_WATER_output

    return unpacked


def stacking_zip_file(filename_zip, path_and_row, image_basename):
    """
        (1) unzipping the downloaded Landsat zip file
        (2) stacking the Landsat image

        Args
        filename_zip
        path_and_row
        image_basename
    """

    # temporary output directory of the zip file, this folder will be deleted after stacking
    output_directory_unzip = join(rootpath, 'data', 'unzip_temp', image_basename)

    if not os.path.exists(output_directory_unzip):
        os.makedirs(output_directory_unzip)

    with tarfile.open(filename_zip) as tar_ref:
        tar_ref.extractall(output_directory_unzip)
    # print('output directory',output_directory_unzip)

    # band settings for Landsat 4, 5,7 and Landsat 8 are different
    if (image_basename[0:4] == 'LE07') | (image_basename[0:4] == 'LT05') | ((image_basename[0:4] == 'LT04')):

        # optical bands
        img_B1 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B1.TIF'))
        img_B2 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B2.TIF'))
        img_B3 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B3.TIF'))
        img_B4 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B4.TIF'))
        img_B5 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B5.TIF'))
        img_B7 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B7.TIF'))

        # thermal bands
        img_B6 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_ST_B6.TIF'))

        # image quality control
        # 0 -- clear; 1 -- water; 2 -- shadow; 3 -- snow; 4 -- cloud; 255 -- fill value
        img_QC = qabitval_array(gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_QA_PIXEL.TIF')))

    else:
        # optical bands
        img_B1 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B2.TIF'))
        img_B2 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B3.TIF'))
        img_B3 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B4.TIF'))
        img_B4 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B5.TIF'))
        img_B5 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B6.TIF'))
        img_B7 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_SR_B7.TIF'))

        # thermal bands
        img_B6 = gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_ST_B10.TIF'))

        # image quality control
        # 0 -- clear; 1 -- water; 2 -- shadow; 3 -- snow; 4 -- cloud; 255 -- fill value
        img_QC = qabitval_array(gdal_array.LoadFile(join(output_directory_unzip, image_basename + '_QA_PIXEL.TIF')))

    ##
    # scale the optical surface reflectance to 0-10000
    img_B1 = 10000 * (img_B1 * 2.75e-05 - 0.2)
    img_B2 = 10000 * (img_B2 * 2.75e-05 - 0.2)
    img_B3 = 10000 * (img_B3 * 2.75e-05 - 0.2)
    img_B4 = 10000 * (img_B4 * 2.75e-05 - 0.2)
    img_B5 = 10000 * (img_B5 * 2.75e-05 - 0.2)
    img_B7 = 10000 * (img_B7 * 2.75e-05 - 0.2)

    # thermal band
    img_B6 = 10 * (img_B6 * 0.00341802 + 149)

    ##
    obj_proj = gdal.Open(join(output_directory_unzip, image_basename + '_SR_B1.TIF'))
    src_proj = obj_proj.GetProjection()
    src_geotrans = obj_proj.GetGeoTransform()

    bands, nrows, ncols = 8, np.shape(img_B1)[0], np.shape(img_B1)[1]
    img_stack_output = np.zeros((bands, nrows, ncols), dtype=np.int16)
    img_stack_output[0, :, :] = img_B1

    # band sequence: blue, green, red, nir, swir1, swir2, thermal, QC
    img_stack_output[0, :, :] = img_B1
    img_stack_output[1, :, :] = img_B2
    img_stack_output[2, :, :] = img_B3
    img_stack_output[3, :, :] = img_B4
    img_stack_output[4, :, :] = img_B5
    img_stack_output[5, :, :] = img_B7
    img_stack_output[6, :, :] = img_B6
    img_stack_output[7, :, :] = img_QC

    # output the stacking file
    path_output = join(rootpath, 'data', 'Level2_stacking', path_and_row)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    filename_output = join(path_output, image_basename + '_stack.tif')

    print('output filename of the stacking file: {}'.format(filename_output))

    obj_save = gdal.GetDriverByName('GTiff').Create(filename_output, ncols, nrows, bands, gdalconst.GDT_Int16)
    obj_save.SetGeoTransform(src_geotrans)
    obj_save.SetProjection(src_proj)

    for i_bands in range(0, bands):
        band = obj_save.GetRasterBand(i_bands + 1)
        band.WriteArray(img_stack_output[i_bands, :, :])

    del obj_save
    del img_B1, img_B2, img_B3, img_B4, img_B5, img_B6, img_B7, img_QC, img_stack_output

    return output_directory_unzip, filename_output

def create_ARD_block(filename_stacking_output, path_and_row, image_basename):
    """
        create the block based on the stacking file, major steps include:
        (1) project the original file to the ARD projection system
        (2) clip the projected file to each tile
        (3) split the results into each block

        Args:
            filename_stacking_output:
            path_and_row:
            image_basename:
    """

    dst_proj = 'PROJCS["Albers_Conic_Equal_Area",GEOGCS["WGS 84",DATUM["WGS_1984",'     \
               'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'     \
               'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],'     \
               'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],'     \
               'PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],'     \
               'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

    obj_stack = gdal.Open(filename_stacking_output)

    # project the original file into the VRT format, determine the boundary of the Landsat image
    params_VRT = gdal.WarpOptions(format='VRT', dstSRS=dst_proj, xRes=RES, yRes=RES,
                                  resampleAlg=gdal.GRIORA_NearestNeighbour, dstNodata=255)
    dst_boundary = gdal.Warp(destNameOrDestDS='', srcDSOrSrcDSTab=obj_stack, options=params_VRT)

    dstboundary_geotrans = dst_boundary.GetGeoTransform()
    x_size, y_size = dst_boundary.RasterXSize, dst_boundary.RasterYSize

    xLeft = dstboundary_geotrans[0]
    yTop = dstboundary_geotrans[3]
    xRight = xLeft + x_size * RES
    yBottom = yTop - y_size * RES

    p_img = Polygon([(xLeft, yBottom), (xLeft, yTop), (xRight, yTop), (xRight, yBottom)])

    path_shp = join(rootpath, 'data', 'tile_grid', 'individual_grid')
    # list_filename_shp = glob.glob(join(path_shp, '*.shp'))   # be careful with the file order

    for i_shp in range(0, 36):

        v_index = 3 - i_shp // 9
        h_index = i_shp % 9
        tilename = 'h' + '%02d' % h_index + 'v' + '%02d' % v_index

        filename_shp = join(path_shp, '{}.shp'.format(tilename))

        # read the shapefile, get the location information
        shp = fiona.open(filename_shp)
        bds = shp.bounds

        ll = (bds[0], bds[1])
        ur = (bds[2], bds[3])
        coords = list(ll + ur)
        xmin_shp, xmax_shp, ymin_shp, ymax_shp = coords[0], coords[2], coords[1], coords[3]

        p_shp = Polygon([(xmin_shp, ymin_shp), (xmin_shp, ymax_shp), (xmax_shp, ymax_shp), (xmax_shp, ymin_shp)])

        if p_img.intersects(p_shp) == True:
            try:
                params = gdal.WarpOptions(format='VRT', dstSRS=dst_proj,
                                          outputBounds=[xmin_shp, ymin_shp, xmax_shp, ymax_shp], xRes=RES, yRes=RES,
                                          resampleAlg=gdal.GRIORA_NearestNeighbour, dstNodata=255)
                dst = gdal.Warp(destNameOrDestDS='', srcDSOrSrcDSTab=obj_stack, options=params)
                img_ard = dst.ReadAsArray()
                del dst

                if (np.unique(img_ard[-1, :, :]) == 255).all() == True:
                    # print('the whole tile is invalid: {}_{}'.format(path_and_row, tilename))
                    pass
                else:
                    logging.info('clip the ARD tile: {}_{}'.format(path_and_row, tilename))
                    # clip the ARD .tif image into different blocks
                    for row_id in range(0, NRow, Block_Size):
                        for col_id in range(0, NCol, Block_Size):

                            block_id = 'row' + '%04d' % row_id + 'col' + '%04d' % col_id

                            img_ard_block = img_ard[:, row_id:row_id + Block_Size, col_id:col_id + Block_Size]

                            if (np.unique(img_ard_block[-1, :, :]) == 255).all() == True:
                                pass
                            else:
                                output_folder_ARD_block = join(outrootpath_scratch, 'data', 'ARD_block', tilename, block_id)
                                if not os.path.exists(output_folder_ARD_block):
                                    os.makedirs(output_folder_ARD_block)

                                output_filename_ARD_block = join(output_folder_ARD_block, '{}_{}_{}.npy'.format(image_basename, tilename, block_id))
                                np.save(output_filename_ARD_block, img_ard_block)

            except Exception as e:
                logging.info('{} clip failed'.format(tilename))
                print(e)
                pass

    return None

def write_finish_flag(path_and_row, image_basename):
    """
        write the stacking finished block .txt file

        Args:
            path_and_row
            image_basename
    """

    outputfolder_blockflag = join(rootpath, 'data', 'blocking_flag', path_and_row)
    if not os.path.exists(outputfolder_blockflag):
        os.makedirs(outputfolder_blockflag, exist_ok=True)
    filaname_blockflag = join(outputfolder_blockflag, '{}.txt'.format(image_basename))
    f = open(filaname_blockflag, 'w')
    f.write('{} blocking finished'.format(image_basename))
    f.close()

    return None


# outrootpath_scratch = r'/shared/cn451/Falu/LCM_diversity'
outrootpath_scratch = r'/scratch/zhz18039/fah20002/LCM_diversity'
list_landsat_collection = ['landsat_tm_c2_l2', 'landsat_etm_c2_l2', 'landsat_ot_c2_l2']

RES = 30
NRow, NCol = 2500, 2500
Block_Size = 250

@click.command()
@click.option('--path_and_row', type=str, default=None, help='path and and row')
@click.option('--rank', type=int, default=0, help='job array id, e.g., 1-200')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')

def main(path_and_row, rank, n_cores):


    logging.basicConfig(filename=join(pwd, '{}.log'.format(path_and_row)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.info('rank {}: stacking running start for {}'.format(rank, path_and_row))

    df_all = get_landsat_df(path_and_row)
    df_task = df_task_obtain(df_all, path_and_row)

    iteration_flag = 0
    while len(df_task)!=0:

        df_task = df_task_obtain(df_all, path_and_row)

        # each core
        num_total_landsat = len(df_task)
        each_core_imagenum = int(np.ceil(num_total_landsat/n_cores))

        for i in range(0,each_core_imagenum):

            new_rank = rank - 1 + i*n_cores
            print(new_rank)
            if new_rank > num_total_landsat -1: # means that all folder has been processed
                print('this is the last running task')
                break

            filename_zip = df_task.loc[df_task.iloc[new_rank]['index'], 'zip_filename']
            image_basename = df_task.loc[df_task.iloc[new_rank]['index'], 'image_basename']

            print('filename for process: {}'.format(filename_zip))
            logging.info('rank {}: filename for process: {}'.format(rank, image_basename))

            filename_output = join(rootpath, 'data', 'Level2_stacking', path_and_row, '{}_stack.tif'.format(image_basename))
            if os.path.exists(filename_output):
                print('the stacking already finished for {}'.format(filename_output))
                logging.info('rank {}: the stacking already finished for {}'.format(rank, image_basename))

                # create the ARD_block
                starttime_blocking = time.perf_counter()
                create_ARD_block(filename_output, path_and_row=path_and_row, image_basename=image_basename)
                endtime_blocking = time.perf_counter()
                logging.info('blocking running time:{}'.format(endtime_blocking - starttime_blocking))

                write_finish_flag(path_and_row, image_basename)

                logging.info('rank {}: the blocking finished for {}'.format(rank, image_basename))

            else:
                starttime_stacking = time.perf_counter()
                path_temp_zip, filename_stacking_output = stacking_zip_file(filename_zip=filename_zip,
                                                                            path_and_row=path_and_row,
                                                                            image_basename=image_basename)
                endtime_stacking = time.perf_counter()
                logging.info('stacking running time:{}'.format(endtime_stacking-starttime_stacking))

                # delete the unzip folder
                shutil.rmtree(path_temp_zip, ignore_errors=True)
                logging.info('rank {}: the stacking finished for {}'.format(rank, image_basename))

                # create the ARD_block
                starttime_blocking = time.perf_counter()
                create_ARD_block(filename_stacking_output, path_and_row=path_and_row,image_basename=image_basename)
                endtime_blocking = time.perf_counter()
                logging.info('blocking running time:{}'.format(endtime_blocking - starttime_blocking))

                write_finish_flag(path_and_row, image_basename)

                logging.info('rank {}: the blocking finished for {}'.format(rank, image_basename))

        if iteration_flag <= 10:
            pass
        else:
            logging.info('rank {}: exceed the maximum iteration round, length of dt_task {}'.format(rank, len(df_task)))
            break

        iteration_flag+=1

    logging.info('rank {}: all task has been done, length of dt_task {}'.format(rank, len(df_task)))



if __name__ == "__main__":
    main()

