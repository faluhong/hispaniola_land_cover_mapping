"""
    calculate the change matrix in the Hispaniola island
"""

import numpy as np
import os
from os.path import join
from osgeo import gdal_array


def change_matrix_generate(array_lc_former, array_lc_latter, landcover_types=8):
    """
        generate the change matrix, compared with confusion_matrix, can ensure 7 types are calculated
    """
    change_matrix = np.zeros((landcover_types, landcover_types), dtype=int)
    for i in range(0, landcover_types):
        for j in range(0, landcover_types):
            mask_tmp = (array_lc_former == i + 1) & (array_lc_latter == j + 1)
            change_matrix[i, j] = np.count_nonzero(mask_tmp)
    return change_matrix


def matrix_normalized(input_matrix):
    """
    normalize the row of the matrix to make each row sums up to 1
    (1) each row value minus the minimum one avoid the negative values
    (2) after step(1), each row was normalized to make each row sums up to 1
    """

    output_matrix = np.zeros((input_matrix.shape), dtype=float)
    for i_row in range(0, np.shape(input_matrix)[0]):
        if np.sum(input_matrix[i_row, :]) == 0:
            # if one land cover type does not exist in the former date, 1 will be assigned to the diagonal value
            # e.g., develop does not exist in 1984, the change_matrix for develop (type 1) will be [0, 0, 0, 0, 0, 0, 0]
            # after the normalization, the transition matrix will be [1, 0, 0, 0, 0, 0, 0]
            # output_matrix[i_row, :] = 0
            output_matrix[i_row, i_row] = 1
        else:
            tmp = input_matrix[i_row, :].copy()
            output_matrix[i_row, :] = tmp / np.sum(tmp)

    return output_matrix


def transition_prob_matrix_two_times(img_lc_t1, img_lc_t2, landcover_types=8):
    """
        calculate the transition probability matrix from time t1 to time t2
    Args:
        img_lc_t1:
        img_lc_t2:
        landcover_types:

    Returns:

    """
    change_matrix_t1_t2 = change_matrix_generate(img_lc_t1, img_lc_t2, landcover_types=landcover_types)
    transition_prob_matrix_t1_t2 = matrix_normalized(change_matrix_t1_t2)

    class change_stats:
        def __init__(self, change_matrix_t1_t2, transition_prob_matrix_t1_t2):
            self.change_matrix = change_matrix_t1_t2
            self.transition_prob_matrix = transition_prob_matrix_t1_t2

        def print_change_matrix(self):
            print(self.change_matrix)
            print(self.transition_prob_matrix)

    output_change = change_stats(change_matrix_t1_t2, transition_prob_matrix_t1_t2)

    return output_change


def land_cover_map_read_published_version(year, country_flag='hispaniola'):
    filename_country_id = join(rootpath, 'data', 'hispaniola_polygon', 'countryid_hispaniola.tif')
    img_country_id = gdal_array.LoadFile(filename_country_id)

    filename = join(rootpath, 'data', 'hispaniola_lc', f'hispaniola_lc_{year}.tif')
    img = gdal_array.LoadFile(filename)
    img = img.astype(float)

    if country_flag == 'hispaniola':
        img[img_country_id == 0] = np.nan
    elif country_flag == 'haiti':
        img[img_country_id != 1] = np.nan
    elif country_flag == 'dr':
        img[img_country_id != 2] = np.nan

    return img


if __name__ == '__main__':

    country_flag = 'haiti'
    # country_flag = 'dr'

    pwd = os.getcwd()
    rootpath = os.path.abspath(os.path.join(pwd, '..'))

    list_year = np.arange(1996, 2023)
    landcover_types = 8

    transition_prob_matrix_adjacent = np.zeros((len(list_year), landcover_types, landcover_types), dtype=float)
    transition_prob_matrix_adjacent[0, :, :] = np.identity(landcover_types)   # the first year is the identity matrix

    transition_prob_matrix_tmp = np.identity(landcover_types)
    for i_year in range(0, len(list_year) - 1):
        year = list_year[i_year]
        print(year)

        img_lc_t1 = land_cover_map_read_published_version(list_year[i_year], country_flag)
        img_lc_t2 = land_cover_map_read_published_version(list_year[i_year + 1], country_flag)

        change_stats_adjacent = transition_prob_matrix_two_times(img_lc_t1, img_lc_t2, landcover_types=landcover_types)
        transition_prob_matrix_adjacent[i_year + 1] = change_stats_adjacent.transition_prob_matrix

    ## output the change matrix
    output_path = join(rootpath, 'results', 'change_matrix')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    output_adjacent_prob_matrix = join(output_path, f'{country_flag}_adjacent_matrix.npy')
    np.save(output_adjacent_prob_matrix, transition_prob_matrix_adjacent)


