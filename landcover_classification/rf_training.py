"""
    Train the random forest model for land cover classification
"""

import time
import joblib
import numpy as np
import os
from os.path import join
from sklearn.ensemble import RandomForestClassifier

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))


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

    # load the training data for the random forest model
    filename_xtraining = join(rootpath, 'data', 'rf_training', 'x_training_refine_i01.npy')
    filename_ytraining = join(rootpath, 'data', 'rf_training', 'y_training_refine_i01.npy')

    x_training = np.load(filename_xtraining)
    y_training = np.load(filename_ytraining)
    print(y_training.shape)

    # random forest training, the training time is about 2 minutes
    rf_classifier = rf_training(x_training, y_training)

    # save the random forest model
    output_folder_rf_classifier = join(rootpath, 'results', 'rf_output')
    if not os.path.exists(output_folder_rf_classifier):
        os.makedirs(output_folder_rf_classifier, exist_ok=True)

    output_filename_rf_classifier = join(output_folder_rf_classifier, 'rf_classifier_i01.joblib')
    joblib.dump(rf_classifier, output_filename_rf_classifier)






