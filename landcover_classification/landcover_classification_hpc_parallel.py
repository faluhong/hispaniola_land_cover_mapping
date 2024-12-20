"""
    conduct the land cover classification in parallel mode in HPC
    The basic idea is conducting the classification for each block in parallel mode.
    After all blocks are classified, merge the land cover and output
"""

import time
import numpy as np
import os
from os.path import join
import sys
import glob
import pandas as pd
import joblib
from osgeo import gdal, gdal_array, gdalconst
import click
import logging
from scipy.ndimage import label, generate_binary_structure
from skimage.morphology import erosion

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))

from landcover_classification.lc_classification import (get_output_rootpath, get_output_path,
                                                        read_dem, read_cold_reccg,
                                                        land_cover_classification,
                                                        img_lcmap_block_acquire, landcover_fill, landcover_merge,
                                                        get_projection_info, landcover_output)

NBands, NRow, NCol = 8, 2500, 2500
Block_Size = 250

record_start_year = 1984
record_end_year = 2022

list_year = np.arange(record_start_year, record_end_year + 1)


def get_all_task():
    """
        get all the task block for running

        Returns:
            df: The dataframe containing the tile name and block name for all COLD reccg blocks
    """
    list_tilename = ['h05v02']

    df = pd.DataFrame(columns=['rank', 'tilename', 'blockname'])
    rank_id = 0
    for i_tile, tilename in enumerate(list_tilename):
        list_filename_cold_reccg = glob.glob(join(rootpath, 'data', 'cold_reccg', tilename, '*.npy'))
        for i_filename, filename_cold in enumerate(list_filename_cold_reccg):
            blockname = os.path.split(filename_cold)[-1][0:14]

            df.loc[rank_id, 'rank'] = rank_id
            df.loc[rank_id, 'tilename'] = tilename
            df.loc[rank_id, 'blockname'] = blockname
            rank_id += 1

    return df


def get_finished_task(landcover_version):
    """
        get the blocks that has been finished
    Args:
        landcover_version:
    Returns:
        df: The dataframe containing the tile name and block name for the finished blocks
    """

    list_finished_tile = os.listdir(join(rootpath, 'results', '{}_landcover_classification'.format(landcover_version)))

    df = pd.DataFrame(columns=['rank', 'tilename', 'blockname'])
    rank_id = 0
    for i_tile, tilename in enumerate(list_finished_tile):

        list_filename_landcover = glob.glob(join(rootpath, 'results', '{}_landcover_classification'.format(landcover_version),
                                                 tilename, 'LC_eachyear_1984_2022', '*.tif'))

        for i_filename, filename_landcover in enumerate(list_filename_landcover):
            blockname = os.path.split(filename_landcover)[-1][7:21]

            if os.path.getsize(filename_landcover) > 0:
                df.loc[rank_id, 'rank'] = rank_id
                df.loc[rank_id, 'tilename'] = tilename
                df.loc[rank_id, 'blockname'] = blockname
                rank_id += 1

    return df


def get_task_block(df_all_block, df_finished_block):
    """
        get the block info for running, i.e., exclude the finished blocks from all blocks

        Args:
            df_all_block: dataframe containing the information of all blocks
            df_finished_block: dataframe containing the information of finished blocks
        Returns:
            The dataframe containing the tile name and block name for task blocks (waiting for running)
    """

    df_running = pd.DataFrame(columns=['rank', 'tilename', 'blockname'])

    rank_id = 0
    for i_rank in range(0, len(df_all_block)):
        tilename = df_all_block.loc[i_rank, 'tilename']
        blockname = df_all_block.loc[i_rank, 'blockname']

        mask_tilename = df_finished_block['tilename'].values == tilename
        mask_blockname = df_finished_block['blockname'].values == blockname

        if np.count_nonzero(mask_tilename & mask_blockname) == 1:
            pass
        else:
            df_running.loc[rank_id, 'rank'] = rank_id
            df_running.loc[rank_id, 'tilename'] = tilename
            df_running.loc[rank_id, 'blockname'] = blockname
            rank_id += 1

    print('number of task block is {}'.format(len(df_running)))

    return df_running


def mosaic_land_cover_map(landcover_version):
    """
        read the land cover block and mosaic for the whole study area
        This function needs to be modified based on your own study sites

        Args:
            landcover_version
        Returns:
            the img array for the whole study area
    """

    list_tilename = ['h05v02']  # change the tile name

    img_lcmap = np.zeros((len(list_year), 500, 750), dtype=np.int8)  # change the shape information based on your own study sites

    for i_tile in range(0, len(list_tilename)):

        tilename = list_tilename[i_tile]

        list_tif_file = glob.glob(join(rootpath, 'results', '{}_landcover_classification'.format(landcover_version), tilename, 'LC_eachyear_1984_2022', '*.tif'))

        for i_file, img_filename in enumerate(list_tif_file):
            blockname = os.path.split(img_filename)[-1][7:21]

            row_id_block = int(blockname[3:7])
            col_id_block = int(blockname[10:14])

            row_id_studyarea = row_id_block - 1000
            col_id_studyarea = col_id_block

            img_lcmap[:, row_id_studyarea:row_id_studyarea + Block_Size, col_id_studyarea:col_id_studyarea + Block_Size] = gdal_array.LoadFile(img_filename)

    return img_lcmap


def mmu_erosion_process(img_lc, cross=np.ones((3, 3)), connection_structure=generate_binary_structure(2, 2)):
    """
        Define the Minimum Mapping Unit (MMU) for the primary forest pixels

        The PF clusters need to have the structure of 3X3 to be considered as the primary forest

        Args:
            img_lc: the land cover map
            cross: the structure of the erosion process, default is 3X3
            connection_structure: the structure of the connected pixels, default is 8-connected
    """

    pf_mask = (img_lc == 2) | (img_lc == 3)  # extract the primary forest mask

    img_core_pf = erosion(pf_mask, cross)  # get the PF pixels that has the 3X3 core area defined with cross

    # using scipy.ndimage.label to get the clustered primary forest
    img_labelled_cluster, num_features = label(pf_mask, structure=connection_structure)  # get the primary forest cluster with each cluster labelled

    label_cluster_with_core_pf = img_labelled_cluster[img_core_pf]  # get the labels of clusters which contain the core primary forest structure

    mask_cluster_with_core_pf = np.isin(img_labelled_cluster, label_cluster_with_core_pf)  # mask to get the PF clusters contain the core PF structure

    img_lc_filter = img_lc.copy()
    img_lc_filter[(pf_mask) & (~mask_cluster_with_core_pf)] = 4  # convert the PF clusters that don't contain the core PF structure into SF

    return img_lc_filter


def mosaic_output(landcover_version, src_geotrans, dst_proj, img_lcmap, year):
    """
        output the mosaic land cover map
        Args:
            landcover_version:
            src_proj: the proj info
            src_geotrans: geolocation info
            img_lcmap: the land cover map
            year:
        Returns:
    """

    ncols, nrows = np.shape(img_lcmap)[1], np.shape(img_lcmap)[0]

    path_output = join(rootpath, 'results', '{}_landcover_classification'.format(landcover_version), 'mosaic')
    if not os.path.exists(path_output):
        os.makedirs(path_output, exist_ok=True)

    filename_output = join(path_output, '{}_{:4d}_landcover.tif'.format(landcover_version, year))

    tif_sat_temp = gdal.GetDriverByName('GTiff').Create(filename_output, ncols, nrows, 1,
                                                        gdalconst.GDT_Byte,
                                                        options=["COMPRESS=LZW"])
    tif_sat_temp.SetGeoTransform(src_geotrans)
    tif_sat_temp.SetProjection(dst_proj)

    Band = tif_sat_temp.GetRasterBand(1)
    Band.WriteArray(img_lcmap)

    del tif_sat_temp

    return filename_output


def add_pyramids_color_in_lc_tif(filename_tif, list_overview=None):
    """
        add pyramids and color table in the tif file
    """

    if list_overview is None:
        list_overview = [2, 4, 8, 16, 32, 64]

    colors = np.array([np.array([255, 255, 255, 255]),    # Ocean
                       np.array([241, 1, 0, 255]),        # Developed
                       np.array([29, 101, 51, 255]),      # Primary wet forest
                       np.array([208, 209, 129, 255]),    # Primary dry forest
                       np.array([108, 169, 102, 255]),    # Secondary forest
                       np.array([174, 114, 41, 255]),     # Shrub/Grass
                       np.array([72, 109, 162, 255]),     # Water
                       np.array([200, 230, 248, 255]),    # Wetland
                       np.array([179, 175, 164, 255]),    # Other
                       ])


    dataset = gdal.Open(filename_tif, gdal.GA_Update)

    # Generate overviews/pyramids
    # The list [2, 4, 8, 16, 32] defines the downsampling factors for the overviews
    dataset.BuildOverviews(overviewlist=list_overview)

    # Get the first band of the image
    band = dataset.GetRasterBand(1)

    # Create a new color table
    color_table = gdal.ColorTable()

    # Set the color for each value in the color table
    for i in range(0, len(colors)):
        color = tuple(colors[i])
        color_table.SetColorEntry(i, color)

    # Assign the color table to the band
    band.SetRasterColorTable(color_table)

    # Save the changes and close the dataset
    dataset = None

    return None

@click.command()
@click.option('--rank', type=int, default=0, help='rank  $SLURM_ARRAY_TASK_ID')
@click.option('--n_cores', type=int, default=1, help='the total applied cores   $SLURM_ARRAY_TASK_MAX')
@click.option('--landcover_version', type=str, default='v1', help='land cover version')
@click.option('--post_processing_flag', type=int, default=1, help='post-processing flag')
def main(rank, n_cores, landcover_version, post_processing_flag):

    if post_processing_flag == 0:
        post_processing_des = 'No rule applied'
    elif post_processing_flag == 1:
        post_processing_des = 'PF rule after 1996 & NBR correction included & Develop correction & last segment rule'
    else:
        post_processing_des = None

    output_rootpath = get_output_rootpath(landcover_version)  # get the root output directory

    logging.basicConfig(filename=join(output_rootpath, '{}.log'.format(landcover_version)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.info('post processing rule: {} {}'.format(post_processing_flag, post_processing_des))
    logging.info('land cover output version {}'.format(landcover_version))

    finished_txt = join(output_rootpath, 'land_cover_classification_finished.txt')  # .txt file to indicate the classification for each block is done

    if rank == 1:  # the first rank is used for merge and output
        while not os.path.exists(finished_txt):  # if
            time.sleep(5)

        logging.info('mosaic, filter, and output the land cover map')

        dst_geotrans = (2708008.6177587425336242, 30.0, 0.0, -140855.6491435449570417, 0.0, -30.0)

        dst_proj = ('PROJCS["Albers_Conic_Equal_Area",GEOGCS["WGS 84",DATUM["WGS_1984",' 
                   'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' 
                   'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],' 
                   'PROJECTION["Albers_Conic_Equal_Area"],' 
                   'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],' 
                   'PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],' 
                   'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')

        # mosaic the output land cover
        img_lcmap = mosaic_land_cover_map(landcover_version)

        for i_year, year in enumerate(list_year):
            img_lcmap_eachyear = img_lcmap[i_year, :, :].copy()

            # using the Minimum Mapping Unit to remove the scattered primary forest pixels
            img_lcmap_eachyear = mmu_erosion_process(img_lcmap_eachyear, cross=np.ones((3, 3)),
                                                     connection_structure=generate_binary_structure(2, 2))
            img_lcmap[i_year, :, :] = img_lcmap_eachyear

            # output the annual land cover map
            filename_output = mosaic_output(landcover_version, dst_geotrans, dst_proj, img_lcmap_eachyear, year)

            # add pyramids and color table in the mosaic tif file
            add_pyramids_color_in_lc_tif(filename_output)

    else:
        offset_flag = 1  # the first rank is used for merge the land cover, an offset is needed to allocate the remaining cores
        while os.path.exists(finished_txt) == False:

            # get the number of tasks for running
            df_all_task = get_all_task()
            df_finish_block = get_finished_task(landcover_version)
            df_task = get_task_block(df_all_task, df_finish_block)

            count_block = len(df_task)
            each_core_task = int(np.ceil(count_block / (n_cores - offset_flag)))  # the number of block that each core for running

            for i in range(0, each_core_task):
                new_rank = rank - (offset_flag + 1) + i * (n_cores - offset_flag)
                if new_rank > count_block - 1:  # means that all folder has been processed
                    print('this is the last running task')
                    break

                # land cover classification for single block
                tilename = df_task.loc[new_rank, 'tilename']
                blockname = df_task.loc[new_rank, 'blockname']

                logging.info('rank {}: classification for tile {} block {}'.format(new_rank, tilename, blockname))

                # read the projection information
                dst_geotrans, src_proj = get_projection_info(tilename)

                # get the output path for the tile
                output_path = get_output_path(landcover_version, tilename)

                # read the pre-trained random forest
                output_name_rf = join(rootpath, 'results', 'rf_output', 'rf_classifier_i01.joblib')
                rf_classifier = joblib.load(output_name_rf)

                # read topography information
                img_dem_block, img_slope_block, img_aspect_block = read_dem(tilename, blockname)

                # read COLD coefficients
                filename_cold_results = join(rootpath, 'data', 'cold_reccg', tilename, '{}_reccg.npy'.format(blockname))
                cold_block = read_cold_reccg(filename_cold_results)

                # land cover classification using random forest, post-processing is applied
                land_cover_classification(rf_classifier, cold_block, img_dem_block, img_slope_block, img_aspect_block, post_processing_flag,
                                          output_path, tilename, blockname)

                # get the land cover images from the temporal segments
                img_lcmap_eachblock = img_lcmap_block_acquire(output_path, tilename, blockname)

                # fill the missing values
                img_lcmap_fill = landcover_fill(img_lcmap_eachblock)

                # merge secondary wet and secondary dry forests; merge barren and cropland to other
                img_lcmap_merge = landcover_merge(img_lcmap_fill)

                # output the land cover map
                landcover_output(img_lcmap_merge, tilename, blockname, dst_geotrans, src_proj, output_path)

            # if all the running task is finished, then write the finish.txt file
            if len(df_task) == 0:
                print('land cover classification finished')

                f = open(finished_txt, 'w')
                f.write('version {} finished'.format(landcover_version))
                f.close()
                logging.info('write the finished txt')


if __name__ == "__main__":
    main()
