"""
    analyze the primary forest landscape metric change
"""

import numpy as np
import os
from os.path import join
import pandas as pd
from osgeo import gdal_array
from scipy.ndimage import label, generate_binary_structure


if __name__ == '__main__':

    pwd = os.getcwd()
    rootpath = os.path.abspath(os.path.join(pwd, '..'))

    # 8-connected
    eight_connected = generate_binary_structure(2, 2)

    img_country_mask = gdal_array.LoadFile(join(rootpath, 'data', 'hispaniola_polygon', 'countryid_hispaniola.tif'))

    df_patch_metric = pd.DataFrame(columns=['year', 'country', 'pf_flag', 'patch_flag', 'value'])

    index = 0
    for year in range(1996, 2023):
        print(year)

        for country_flag in ['haiti', 'dr']:
            if country_flag == 'haiti':
                country_mask = 1
            else:
                country_mask = 2

            img_lc = gdal_array.LoadFile(join(rootpath, 'data', 'hispaniola_lc', f'hispaniola_lc_{year}.tif'))
            img_lc[img_country_mask != country_mask] = 0

            for pf_flag in [2, 3]:
                if pf_flag == 2:
                    pf_des = 'wet'
                else:
                    pf_des = 'dry'

                pf_mask = img_lc == pf_flag  # extract the primary wet forest mask

                # using scipy.ndimage.label to get the clustered primary forest
                img_labelled_cluster, num_features = label(pf_mask, structure=eight_connected)

                # get the count of each cluster size
                patch_label, patch_counts = np.unique(img_labelled_cluster, return_counts=True)
                patch_label_cal = patch_label[1::]   # exclude the non-primary-forest pixel
                patch_counts_cal = patch_counts[1::]  # exclude the non-primary-forest pixel

                patch_count = len(patch_label_cal)
                mean_patch_size = np.nanmean(patch_counts_cal)

                for patch_flag in ['count', 'mean_size']:

                    df_patch_metric.loc[index, 'year'] = year
                    df_patch_metric.loc[index, 'country'] = country_flag
                    df_patch_metric.loc[index, 'pf_flag'] = pf_des
                    df_patch_metric.loc[index, 'patch_flag'] = patch_flag

                    if patch_flag == 'count':
                        df_patch_metric.loc[index, 'value'] = patch_count
                    else:
                        df_patch_metric.loc[index, 'value'] = mean_patch_size

                    index += 1

    df_patch_metric.to_excel(join(rootpath, 'results', 'pf_landscape_metrix.xlsx'))