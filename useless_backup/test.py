"""

Landsat download function

usgsm2m library is required, please refer to the link given below for installation

Reference link
https://pypi.org/project/usgsm2m/
https://github.com/ashutoshkumarjha/usgsm2m
"""

import click
import time
import numpy as np
import os
from os.path import join
import sys
from usgsm2m.api import API
from usgsm2m.usgsm2m import USGSM2M
import fiona
from shapely.geometry.polygon import Polygon
import logging

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

username, password = 'falu_hong', 'University4.'  # USGS account, you can replace with your own username and password

# Initialize a new API instance and get an access key
api = API(username, password)
ee = USGSM2M(username, password)

list_landsat_collection = ['landsat_tm_c2_l2', 'landsat_etm_c2_l2', 'landsat_ot_c2_l2']


def get_product_id(i_collect, centroid_y, centroid_x, start_date, end_date):
    """
    getting the product id of all available Landsat data

    Args:
        centroid_y: Latitude
        centroid_x: Longitude

    Returns:
        list_productid: the list of product id
    """

    list_productid = []

    if i_collect in [1, 2, 3]:
        dataset = list_landsat_collection[i_collect - 1]
        scenes = api.search(
            dataset=dataset,
            latitude=centroid_y,
            longitude=centroid_x,
            start_date=start_date,
            end_date=end_date,
            max_results=10000)

        for i_scenes in range(0, len(scenes)):
            product_id = scenes[i_scenes]['display_id']
            list_productid.append(product_id)

    else:
        for i_temp in range(0, len(list_landsat_collection)):
            dataset = list_landsat_collection[i_temp]
            scenes = api.search(
                dataset=dataset,
                latitude=centroid_y,
                longitude=centroid_x,
                start_date=start_date,
                end_date=end_date,
                max_results=10000)

            for i_scenes in range(0, len(scenes)):
                product_id = scenes[i_scenes]['display_id']
                list_productid.append(product_id)

    return list_productid


def get_path_output(product_id, path_downloadoutput_root):
    """
        getting the dataset and output path from the product id

        Args:
            product_id

        Returns:
            dataset: the Landsat collection name
            path_output: the output path

        Raises:
            No corresponding dataset and output path are found
    """

    pathrow_info = product_id[10:16]
    if product_id[0:4] == 'LT05':
        dataset = list_landsat_collection[0]
        path_output = join(path_downloadoutput_root, dataset, pathrow_info)
    elif product_id[0:4] == 'LT04':
        dataset = list_landsat_collection[0]
        path_output = join(path_downloadoutput_root, dataset, pathrow_info)
    elif product_id[0:4] == 'LE07':
        dataset = list_landsat_collection[1]
        path_output = join(path_downloadoutput_root, dataset, pathrow_info)
    elif product_id[0:4] == 'LC08':
        dataset = list_landsat_collection[2]
        path_output = join(path_downloadoutput_root, dataset, pathrow_info)
    elif product_id[0:4] == 'LC09':
        dataset = list_landsat_collection[2]
        path_output = join(path_downloadoutput_root, dataset, pathrow_info)
    else:
        dataset = None
        path_output = None
        logging.info('No corresponding dataset and path_output are defined {}'.format(product_id))

    if dataset != None:
        if not os.path.exists(path_output):
            os.makedirs(path_output)

    return dataset, path_output


def landsat_download(list_product_id, path_downloadoutput_root):
    """
        download the Landsat data

        Args:
            list_product_id

        Returns:
            list_failed_filename: the list of failed product id
    """

    list_failed_filename = []
    for i_product in range(0, len(list_product_id)):

        product_id = list_product_id[i_product]
        dataset, path_output = get_path_output(product_id, path_downloadoutput_root)

        try:
            start_time = time.perf_counter()
            logging.info('start downaload: {}'.format(product_id))
            ee.download([product_id], dataset=dataset, output_dir=path_output)
            end_time = time.perf_counter()
            logging.info('downloading done: {}'.format(product_id))
            logging.info('downloading time: {}'.format(end_time - start_time))
        except Exception:
            list_failed_filename.append(product_id)
            logging.info('failed download: {}'.format(product_id))

    return list_failed_filename


def redownload_failed_file(list_failed_filename, path_downloadoutput_root):
    """
        redownload the failed Landsat data, iterating for 10 times

        Args:
            list_failed_filename

        Returns:
            success_flag: 1 means successful download all the Landsat data, 0 means failed download for some files
    """

    success_flag = 1

    if len(list_failed_filename) == 0:
        return success_flag
    else:
        count = 0
        while len(list_failed_filename) > 0:

            for i_redownload in range(0, len(list_failed_filename)):

                product_id = list_failed_filename[i_redownload]
                dataset, path_output = get_path_output(product_id, path_downloadoutput_root)

                try:
                    start_time = time.perf_counter()
                    logging.info('start downaload: {}'.format(product_id))
                    ee.download([product_id], dataset=dataset, output_dir=path_output)
                    end_time = time.perf_counter()
                    logging.info('downloading done: {}'.format(product_id))
                    list_failed_filename.append(product_id)
                    logging.info('downloading time: {}'.format(end_time - start_time))
                except Exception:
                    logging.info('failed download: {}'.format(product_id))

            count += 1

            if count > 10:
                logging.info('maximum download iteration')
                for _ in list_failed_filename:
                    logging.info(_)

                success_flag = 0
                break

    return success_flag


# @click.command()
# @click.option('--i_collect', type=int, default=1, help='the landsat collection id, 1-landsat_tm_c2_l2, 2-landsat_etm_c2_l2, 3-landsat_ot_c2_l2, 4-all collections')
# @click.option('--path_and_row', type=str, default=None, help='the landsat path and row id, e.g, 008046')
# @click.option('--start_date', type=str, default='1980-01-01')
# @click.option('--end_date', type=str, default='2023-05-01')
# def main(i_collect, path_and_row, start_date, end_date):

if __name__ == "__main__":

    i_collect = 3
    path_and_row = '010047'
    start_date = '2023-03-01'
    end_date = '2023-04-01'

    filename_wrs = join(rootpath, 'data', 'WRS2_descending_0', 'WRS2_descending.shp')
    shp_wrs = fiona.open(filename_wrs)

    path_download = join(rootpath, 'data', 'Level2')
    if not os.path.exists(path_download):
        os.makedirs(path_download, exist_ok=True)

    path_logging = join(path_download, 'download_log')
    if not os.path.exists(path_logging):
        os.makedirs(path_logging, exist_ok=True)

    logging.basicConfig(filename=join(path_logging, '{}_{}.log'.format(i_collect, path_and_row)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    logging.info('i_collect {}'.format(i_collect))
    logging.info('path_and_row {}'.format(path_and_row))
    logging.info('start date {}'.format(start_date))
    logging.info('end date {}'.format(end_date))
    logging.info('download path {}'.format(path_download))
    logging.info('logging path {}'.format(join(path_logging, '{}_{}.log'.format(i_collect, path_and_row))))

    for i, record in enumerate(shp_wrs):
        path_and_row_shp = '%03d%03d' % (record['properties']['PATH'], record['properties']['ROW'])

        if path_and_row_shp == path_and_row:

            coordinates = record['geometry']['coordinates'][0]
            coordinates_polygon = Polygon(coordinates)

            # central latitude and longitude of the selected PathRow
            centroid_x, centroid_y = coordinates_polygon.centroid.coords.xy[0][0], coordinates_polygon.centroid.coords.xy[1][0]

            list_product_id = get_product_id(i_collect, centroid_y, centroid_x, start_date, end_date)

            list_failed_filename = landsat_download(list_product_id, path_download)

            success_flag = redownload_failed_file(list_failed_filename, path_download)

            if success_flag == 1:
                logging.info('download complete for {}'.format(path_and_row_shp))
            elif success_flag == 0:
                logging.info('download incomplete for {}'.format(path_and_row_shp))

    ee.logout()


# if __name__ == "__main__":
#     main()