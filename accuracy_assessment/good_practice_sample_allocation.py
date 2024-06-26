"""
    generate the validation sample using the stratified random sample following the good practice strategy.
"""

import numpy as np
import numpy.typing as npt


def total_sample_num_calculate(standar_error_est_overally_accu: float,
                               array_weight: npt.NDArray or float,
                               conjecture_user_accuracy: npt.NDArray or float):

    """calculate the total sample number for the stratified

    Args:
        standar_error_est_overally_accu (float): _description_
        array_weight (npt.NDArray or float): the area weight
        conjecture_user_accuracy (npt.NDArray or float):

    Returns:
        total_sample_num: total selected sample number
    """

    std_stratum = np.sqrt(conjecture_user_accuracy * (1 - conjecture_user_accuracy))
    total_sample_num = np.square(np.nansum(array_weight * std_stratum) / standar_error_est_overally_accu)

    total_sample_num = round(total_sample_num)

    return total_sample_num


def sample_allocation(total_sample_num, array_weight, rare_class_num=100, rare_class_threshold=0.1):
    """
        allocate the sample number for each class

    Args:
        total_sample_num: total sample number
        array_weight: weight for each class
        rare_class_num: the sample number allocated to the rare class
        rare_class_threshold: the weight threshold to determine the rare class

    Returns:
        array_selected_num: the selected sample number of each class (strata)
    """

    array_selected_num = np.zeros((len(array_weight)), dtype=int)

    for i_class, array_weight_class in enumerate(array_weight):
        # allocate the sample number for rare class
        if array_weight_class <= rare_class_threshold:
            array_selected_num[i_class] = rare_class_num

    # for the rest strata, allocate the sample number based on the area proportion
    rest_proportion = array_weight[array_selected_num == 0]
    rest_proportion_redistribute = rest_proportion / np.nansum(rest_proportion)

    rest_sample_count = (total_sample_num - np.nansum(array_selected_num)) * rest_proportion_redistribute
    rest_sample_count = np.round(rest_sample_count)

    array_selected_num[array_selected_num == 0] = rest_sample_count

    return array_selected_num


if __name__ == '__main__':

    # Deforestation, forest gain, stabel forest, stable non forest
    array_weight = np.array([0.02, 0.015, 0.320, 0.645], dtype=float)

    standar_error_est_overall_accu = 0.01
    conjecture_user_accuracy = np.array([0.70, 0.60, 0.90, 0.95], dtype=float)

    total_sample_num = total_sample_num_calculate(standar_error_est_overall_accu, array_weight, conjecture_user_accuracy)

    array_selected_num = sample_allocation(total_sample_num, array_weight, rare_class_num=50, rare_class_threshold=0.1)

    print(total_sample_num, array_selected_num)

