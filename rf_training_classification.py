"""
    An example of training the random forest model and land cover classification
"""

import time
import joblib
import numpy as np
import os
from os.path import join
import sys
from sklearn.ensemble import RandomForestClassifier

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

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


if __name__ == "__main__":

    # load the training data.
    # This is training data to get the random forest I provide to you. The "rf_classifier_i01.joblib" model
    filename_xtraining = join(rootpath, 'data', 'rf_training', 'x_training_refine_i01.npy')
    filename_ytraining = join(rootpath, 'data', 'rf_training', 'y_training_refine_i01.npy')

    x_training = np.load(filename_xtraining)
    y_training = np.load(filename_ytraining)  # around 25083 training samples

    # random forest training, the training time is about 2 minutes
    rf_classifier = rf_training(x_training, y_training)

    # save the random forest model
    output_folder_rf_classifier = join(rootpath, 'results', 'rf_output')
    if not os.path.exists(output_folder_rf_classifier):
        os.makedirs(output_folder_rf_classifier, exist_ok=True)

    output_filename_rf_classifier = join(output_folder_rf_classifier, 'rf_classifier_i01.joblib')
    joblib.dump(rf_classifier, output_filename_rf_classifier)

    ##
    # land cover classification after you get the random forest model
    # The following part is the same as the main function in landcover_classification.py file
    from landcover_classification import (read_dem, read_cold_reccg, land_cover_classification, img_lcmap_block_acquire,
                                          landcover_fill, landcover_merge, get_output_rootpath, get_output_path,
                                          land_cover_plot)

    tilename = 'h05v02'
    blockname = 'row1000col0000'
    post_processing_flag = 0
    landcover_version = 'v3'

    output_rootpath = get_output_rootpath(landcover_version)  # get the root output directory
    output_path = get_output_path(landcover_version, tilename)  # get the output path for the tile

    img_dem_block, img_slope_block, img_aspect_block = read_dem(tilename, blockname)  # read topography information

    filename_cold_results = join(rootpath, 'data', 'cold_reccg', tilename, '{}_reccg.npy'.format(blockname))
    cold_block = read_cold_reccg(filename_cold_results)  # read COLD coefficients

    # land cover classification using random forest, post-processing is applied
    land_cover_classification(rf_classifier, cold_block, img_dem_block, img_slope_block, img_aspect_block, post_processing_flag,
                              output_path, tilename, blockname)

    # get the land cover images from the temporal segments
    img_lcmap_eachblock = img_lcmap_block_acquire(output_path, tilename, blockname)

    # fill the missing values
    img_lcmap_fill = landcover_fill(img_lcmap_eachblock)

    # merge secondary wet and secondary dry forests to secondary forest
    img_lcmap_merge = landcover_merge(img_lcmap_fill)

    # plot the land cover
    land_cover_plot(img_lcmap_merge[-1], title='land cover in 2022')





