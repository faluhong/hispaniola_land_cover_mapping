"""
    Good practice function for the accuracy assessment
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def generate_good_practice_matrix(data, array_weight, array_count, confidence_interval=1.96):
    """
        generate the error matrix following the "Good Practice" strategy
        The validation sample design follows the stratified random sample

        Steps:
        (1) calculate the proportion of area for each cell
        (2) calculate the user's and producer's accuracy
        (3) calculate the variance of user's, producer's and overall accuracy
        (4) reorder the output dataframe

        Ref: Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E., & Wulder, M. A. (2014).
        Good practices for estimating area and assessing accuracy of land change. Remote sensing of Environment, 148, 42-57.
        https://www.sciencedirect.com/science/article/pii/S0034425714000704

        Args:
            data: 2-D array, the error matrix of sample counts
                  ROW represents the map class, COLUMN represents the reference class
            array_weight: weight of each class
            array_count: pixel count of each class
            confidence_interval: value representing the confidence interval, default is 1.96, indicating the 5-95% interval
        Returns:
            df_error_adjust: error matrix based on the proportions of area
    """

    landcover_types = len(data)
    list_landcover = list(np.arange(1, 1 + landcover_types))  # list to indicate the land cover types

    df_confusion = pd.DataFrame(data=data, columns=list_landcover, index=list_landcover)

    df_err_adjust = pd.DataFrame(df_confusion.iloc[0:landcover_types, 0:landcover_types], index=list_landcover, columns=list_landcover)

    df_err_adjust.loc[:, 'n_count'] = df_err_adjust.sum(axis=1)
    df_err_adjust.loc['n_count', :] = df_err_adjust.sum(axis=0)

    df_err_adjust.loc[:, 'UA'] = np.nan
    df_err_adjust.loc['PA', :] = np.nan

    df_err_adjust.loc[list_landcover, 'weight'] = array_weight  # assign weight

    for i_row in range(0, landcover_types):
        df_err_adjust.iloc[i_row, 0: landcover_types] = df_err_adjust.iloc[i_row, 0: landcover_types] / \
                                                        df_err_adjust.loc[i_row + 1, 'n_count'] * df_err_adjust.loc[i_row + 1, 'weight']
        df_err_adjust.loc[i_row + 1, 'total'] = np.nansum(df_err_adjust.iloc[i_row, 0: landcover_types])

    for i_col in range(0, landcover_types):
        df_err_adjust.loc['total', i_col + 1] = np.nansum(df_err_adjust.iloc[0: landcover_types, i_col])

    # calculate the user's and producer's accuracy
    for i in range(0, landcover_types):
        df_err_adjust.loc['PA', i + 1] = df_err_adjust.iloc[i, i] / df_err_adjust.loc['total', i + 1]
        df_err_adjust.loc[i + 1, 'UA'] = df_err_adjust.iloc[i, i] / df_err_adjust.loc[i + 1, 'total']

    df_err_adjust.loc['total', 'total'] = 1

    # calculate the overall accuracy
    overall_accuracy = np.nansum(np.diag(df_err_adjust.iloc[0: landcover_types, 0: landcover_types].values))
    df_err_adjust.loc['PA', 'UA'] = overall_accuracy

    # calculate the user's accuracy variance
    user_accuracy = df_err_adjust['UA'].values[0: landcover_types]
    variance_user_accuracy = user_accuracy * (1 - user_accuracy) / (df_err_adjust['n_count'].values[0: landcover_types] - 1)
    std_user_accuracy = np.sqrt(variance_user_accuracy)

    # calculate the overall accuracy variance
    variance_overall_accuracy = np.power(array_weight, 2) * user_accuracy * (1 - user_accuracy) / (df_err_adjust['n_count'].values[0: landcover_types] - 1)
    variance_overall_accuracy = np.sum(variance_overall_accuracy)
    std_overall_accuracy = np.sqrt(variance_overall_accuracy)

    # calculate the producer's accuracy variance
    producer_accuracy = df_err_adjust.loc['PA', :][0: landcover_types].values
    variance_producer_accuracy = np.zeros(np.shape(producer_accuracy), dtype=float)

    for j in range(0, landcover_types):

        # calculate the estimated Nj
        N_j_estimated = 0
        for i in range(0, landcover_types):
            N_j_estimated = N_j_estimated + array_count[i] / df_confusion.sum(axis=1).values[i] * df_confusion.iloc[i, j]

        # part 1
        part_1 = np.power(array_count[j] * (1 - producer_accuracy[j]), 2) * user_accuracy[j] * (1 - user_accuracy[j])
        n_j = df_confusion.sum(axis=1).values[j]
        part_1 = part_1 / (n_j - 1)

        # part 2
        part_2 = 0
        for i in range(0, landcover_types):
            if i == j:
                pass
            else:
                n_i = df_confusion.sum(axis=1).values[i]
                n_ij = df_confusion.iloc[i, j]

                tmp_1 = np.power(array_count[i], 2) / n_i * n_ij
                tmp_2 = (1 - n_ij / n_i) / (n_i - 1)

                part_2 = part_2 + tmp_1 * tmp_2

        part_2 = np.power(producer_accuracy[j], 2) * part_2

        # final variance producer accuracy
        variance_producer_accuracy[j] = (part_1 + part_2) / np.power(N_j_estimated, 2)

    # standard deviation of producer's accuracy
    std_producer_accuracy = np.sqrt(variance_producer_accuracy)

    # assign the uncertainty to the output table
    df_err_adjust.loc[:, 'UA_uncertainty'] = np.nan
    df_err_adjust['UA_uncertainty'].values[0: landcover_types] = std_user_accuracy * confidence_interval

    df_err_adjust.loc['PA_uncertainty', :] = np.nan
    df_err_adjust.loc['PA_uncertainty', 1: landcover_types] = std_producer_accuracy * confidence_interval

    df_err_adjust.loc['PA_uncertainty', 'UA_uncertainty'] = std_overall_accuracy * confidence_interval

    # reorder the row and columns
    df_err_adjust = df_err_adjust[list_landcover + ['total', 'UA', 'UA_uncertainty', 'n_count', 'weight']]
    df_err_adjust = df_err_adjust.reindex(list_landcover + ['total', 'PA', 'PA_uncertainty', 'n_count'])

    return df_err_adjust


def get_adjusted_area_and_margin_of_error(data, df_err_adjust, array_count, confidence_interval=1.96):
    """
        Calculate the adjusted area and margin of error based on the error matrix generated by the Good Practice strategy
    Args:
        data:
        df_err_adjust:
        array_count:
        confidence_interval:

    Returns:

    """
    stratum_count = df_err_adjust.shape[0] - 4

    # landcover_types = len(data)
    list_stratum_id = list(np.arange(1, 1 + stratum_count))  # list to indicate the land cover types
    df_confusion = pd.DataFrame(data=data, columns=list_stratum_id, index=list_stratum_id)

    array_weight = df_err_adjust['weight'].values[0: stratum_count]
    array_weight = array_weight.astype(float)

    array_mapped_area = df_err_adjust.loc[:, 'total'].values[0: df_err_adjust.shape[0] - 4]
    array_adjusted_area = df_err_adjust.loc['total', :].values[0: df_err_adjust.shape[1] - 5]

    #  get the area in km2
    array_mapped_area = array_mapped_area * 900 / 1000000 * np.nansum(array_count)
    array_adjusted_area = array_adjusted_area * 900 / 1000000 * np.nansum(array_count)

    array_mapped_area = array_mapped_area.astype(float)
    array_adjusted_area = array_adjusted_area.astype(float)

    print(f'mapped area (km2): {np.round(array_mapped_area, 2)}')
    print(f'adjusted area (km2): {np.round(array_adjusted_area, 2)}')
    print(f'map bias: map area - adjusted area {np.round(array_mapped_area - array_adjusted_area, 2)}')

    stand_error_area_proportion = np.zeros(stratum_count, dtype=float)
    for k in range(0, stratum_count):
        sum = 0
        for i in range(0, stratum_count):
            ni = np.nansum(df_confusion.iloc[i, 0:stratum_count].values)
            pik = df_err_adjust.iloc[i, k]

            sum += (array_weight[i] * pik - pik * pik) / (ni - 1)  # Eq.10 in Good Practice paper

        stand_error_area_proportion[k] = np.sqrt(sum)

    print(f'standard_area_proportion {stand_error_area_proportion}')

    stand_error_area = stand_error_area_proportion * np.nansum(array_count) * 900 / 1000000 * confidence_interval
    print(f'standard error of area (km2) estimation at 95% confidence level {np.round(stand_error_area, 2)}')


def plot_df_confusion(array, stratum_des=None, figsize=(16, 13),
                      title=None, x_label='Reference', y_label='Map'):
    """
        plot the confusion matrix
    """

    if stratum_des is None:
        stratum_des = {'1': 'Deforestation',
                       '2': 'Forest gain',
                       '3': 'Stable forest',
                       '4': 'Stable non forest',
                       }

    df_cm = pd.DataFrame(array, index=stratum_des.values(), columns=stratum_des.values())

    figure, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    cmap = matplotlib.cm.GnBu

    tick_labelsize = 16
    axis_labelsize = 20
    title_size = 24
    annosize = 18
    ticklength = 4
    axes_linewidth = 1.5

    im = sns.heatmap(df_cm, annot=True, annot_kws={"size": annosize}, fmt='d', cmap=cmap, cbar=False)
    im.figure.axes[-1].yaxis.set_tick_params(labelsize=annosize)

    ax.tick_params('y', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, left=True, which='major', rotation=0)
    ax.tick_params('x', labelsize=tick_labelsize, direction='out', length=ticklength, width=axes_linewidth, top=False, which='major', rotation=15)

    ax.set_xlabel(x_label, size=axis_labelsize)
    ax.set_ylabel(y_label, size=axis_labelsize)

    overall_accuracy = np.trace(df_cm.values) / np.sum(df_cm.values)
    ax.set_title(f'{title}:  {np.trace(df_cm.values)}/{df_cm.values.sum()}={np.round(overall_accuracy*100, 1)}', size=title_size)

    plt.tight_layout()
    plt.show()


# def main():
if __name__=='__main__':

    # example in "Good Practice" paper
    array_weight = np.array([0.02, 0.015, 0.320, 0.645], dtype=float)
    array_count = np.array([200000, 150000, 3200000, 6450000], dtype=float)

    data = np.array([[66, 0, 5, 4],
                     [0, 55, 8, 12],
                     [1, 0, 153, 11],
                     [2, 1, 9, 313]], dtype=int)

    df_err_adjust = generate_good_practice_matrix(data, array_weight, array_count, confidence_interval=1.96)

    get_adjusted_area_and_margin_of_error(data, df_err_adjust, array_count, confidence_interval=1.96)

    stratum_des = {'1': 'Deforestation',
                   '2': 'Forest gain',
                   '3': 'Stable forest',
                   '4': 'Stable non forest',
                   }

    plot_df_confusion(data, stratum_des=stratum_des, figsize=(10, 8),
                      title=None,
                      x_label='Reference', y_label='Map',)
