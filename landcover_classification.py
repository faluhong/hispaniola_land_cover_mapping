"""
land cover classification from the COLD reccg file to the land cover classification outputs
random forest model is pre-trained
"""

import numpy as np
import os
from os.path import join
import sys
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import json as js
from osgeo import gdal, gdal_array, gdalconst
import matplotlib
import heapq
import fiona
import logging
import seaborn as sns
import matplotlib.patches as mpatches

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))
path_pythoncode = join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

Block_Size = 250

record_start_year = 1984
record_end_year = 2022

list_year = np.arange(record_start_year, record_end_year + 1)

def datetime_to_datenum(dt):
    """convert the datetime to datenum

    Args:
        dt: input datetime
    Returns:
        datenum
    """
    python_datenum = dt.toordinal()
    return python_datenum

def datenum_to_datetime(datenum):
    """
        convert the datenum to datetime
    """
    python_datetime = datetime.fromordinal(int(datenum))
    return python_datetime


def get_projection_info(tilename):
    """
    getting the projection info from the shapefile, including: (1) the proj info; (2) geo_trans
    Args:
        tilename
    Returns:
        src_geotrans: geolocation info
        src_proj: the proj info
    """

    filename_shp_ref = join(rootpath, 'data', 'tile_grid', 'individual_grid', '{}.shp'.format(tilename))

    shp_ref = fiona.open(filename_shp_ref)
    bounds = shp_ref.bounds

    src_proj = 'PROJCS["Albers_Conic_Equal_Area",GEOGCS["WGS 84",DATUM["WGS_1984",' \
               'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
               'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],' \
               'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],' \
               'PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],' \
               'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    src_geotrans = (bounds[0], 30, 0, bounds[3], 0, -30)

    return src_geotrans, src_proj


def get_output_rootpath(output_version):
    """
    creating the output folder
    Args:
        output_version
    Returns:
        output_path
    """

    output_path = join(rootpath, 'results', '{}_landcover_classification'.format(output_version))
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    return output_path


def get_output_path(output_version, tilename):
    """
    creating the output folder for each tile
    Args:
        tilename
        output_version
    Returns:
        output_path
    """

    output_path = join(rootpath, 'results', '{}_landcover_classification'.format(output_version), tilename)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    return output_path


def read_dem(tilename, blockname):
    """
        read the DEM, slope, and aspect for the block
        Args:
            tilename
            blockname
        Returns:
            img_dem_block, img_slope_block, img_aspect_block
    """

    filename_dem = join(rootpath, 'data', 'topography', 'dem_{}.tif'.format(tilename))
    filename_slope = join(rootpath, 'data', 'topography', 'slope_degree_{}.tif'.format(tilename))
    filename_aspect = join(rootpath, 'data', 'topography', 'aspect_{}.tif'.format(tilename))

    img_dem = gdal_array.LoadFile(filename_dem)
    img_slope = gdal_array.LoadFile(filename_slope)
    img_aspect = gdal_array.LoadFile(filename_aspect)

    row_id_intile = int(blockname[3:7])
    col_id_intile = int(blockname[10:14])
    img_dem_block = img_dem[row_id_intile:row_id_intile + Block_Size, col_id_intile:col_id_intile + Block_Size]
    img_slope_block = img_slope[row_id_intile:row_id_intile + Block_Size, col_id_intile:col_id_intile + Block_Size]
    img_aspect_block = img_aspect[row_id_intile:row_id_intile + Block_Size, col_id_intile:col_id_intile + Block_Size]

    return img_dem_block, img_slope_block, img_aspect_block


def prepare_xdata(cold_reccg):
    """
        extract the predictor variables from COLD reccg file
        Args:
            cold_reccg: COLD rec_cg file
        Returns:
            x_data: the prepared x data
    """

    array_t_start = cold_reccg['t_start']
    array_t_end = cold_reccg['t_end']

    array_a0 = cold_reccg['coefs'][:, :, 0]
    array_c1 = cold_reccg['coefs'][:, :, 1] / 10000
    array_central_surfaceref = array_a0 + array_c1 * 0.5 * (np.tile(array_t_start, (7, 1)).T + np.tile(array_t_end, (7, 1)).T)

    array_harmonic = cold_reccg['coefs'][:, :, 2:8].reshape(len(cold_reccg), 42)
    array_rmse = cold_reccg['rmse']

    x_data = np.concatenate((array_central_surfaceref.T, array_c1.T, array_harmonic.T, array_rmse.T)).T

    return x_data


def read_cold_reccg(filename_cold_results):
    """
        read the COLD rec_cg file
        Invalid values needs to be filtered
    Args:
        filename_cold_results:

    Returns:

    """
    cold_block = np.load(join(filename_cold_results))
    x_trainingdata = prepare_xdata(cold_block)

    # the original rec_cg could contain NaN, inf, -inf value, those rec_cg should be filtered
    mask_valid = np.ones(len(cold_block), dtype=bool)
    mask_valid[np.unique(np.where(np.isnan(x_trainingdata))[0])] = False

    cold_block = cold_block[mask_valid]

    return cold_block


def prepare_xdata_with_topography(cold_reccg, img_dem_block, img_slope_block, img_aspect_block):
    """
        prepare the predictor variables incorporating COLD coefficient and topography information (elevation, slope, and aspect information)
        The structure of the predictor variables includes
        (1) central overall surface reflectance of the temporal segment
        (2) slope
        (3) harmonic coefficient
        (4) RMSE
        (5) elevation, slope, aspect

        Args:
            cold_reccg
            img_dem_block
            img_slope_block
            img_aspect_block
        Returns:
            x_data
    """

    array_t_start = cold_reccg['t_start']
    array_t_end = cold_reccg['t_end']

    array_a0 = cold_reccg['coefs'][:, :, 0]
    array_c1 = cold_reccg['coefs'][:, :, 1] / 10000
    array_central_surfaceref = array_a0 + array_c1 * 0.5 * (np.tile(array_t_start, (7, 1)).T + np.tile(array_t_end, (7, 1)).T)

    array_harmonic = cold_reccg['coefs'][:, :, 2:8].reshape(len(cold_reccg), 42)
    array_rmse = cold_reccg['rmse']

    array_pos = cold_reccg['pos'] - 1

    array_col_id_inblock = array_pos % Block_Size
    array_row_id_inblock = (array_pos - array_col_id_inblock) // Block_Size

    array_dem = img_dem_block[array_row_id_inblock, array_col_id_inblock]
    array_dem = array_dem.reshape(len(array_dem), 1)

    array_slope = img_slope_block[array_row_id_inblock, array_col_id_inblock]
    array_slope = array_slope.reshape(len(array_slope), 1)

    array_aspect = img_aspect_block[array_row_id_inblock, array_col_id_inblock]
    array_aspect = array_aspect.reshape(len(array_aspect), 1)

    x_data = np.concatenate((array_central_surfaceref.T, array_c1.T, array_harmonic.T, array_rmse.T,
                             array_dem.T, array_slope.T, array_aspect.T)).T

    return x_data


def land_cover_classification(rf_classifier, cold_block, img_dem_block, img_slope_block, img_aspect_block, post_processing_flag,
                              output_path, tilename, blockname):
    if len(cold_block) == 0:
        logging.info('no available valid data for tile: {} block: {}'.format(tilename, blockname))
    else:
        # prepare the predictor variables for classification
        x_trainingdata_withdem = prepare_xdata_with_topography(cold_block, img_dem_block, img_slope_block, img_aspect_block)

        y_prediction = rf_classifier.predict(x_trainingdata_withdem)  # predict the land cover using random forest
        y_prediction_prob = rf_classifier.predict_proba(x_trainingdata_withdem)  # get the prediction probability

        # refine the classification results
        array_pos = cold_block['pos']
        array_t_start = cold_block['t_start']
        array_t_end = cold_block['t_end']

        for pos_id in range(1, Block_Size * Block_Size + 1):

            if np.count_nonzero(array_pos == pos_id) <= 1:
                pass
            else:
                location_pos = np.where(array_pos == pos_id)[0]
                location_t_start = array_t_start[array_pos == pos_id]
                location_t_end = array_t_end[array_pos == pos_id]

                # post-process the classification results output by random forest
                if post_processing_flag == 0:
                    pass
                else:
                    y_prediction = post_processing(y_prediction, y_prediction_prob, location_pos, location_t_end, location_t_start)

        # prepare the dataset to output the classification results
        array_t_start = cold_block['t_start']
        array_t_end = cold_block['t_end']
        array_pos = cold_block['pos']
        array_LC = y_prediction

        df_predict_each_rec_cg = pd.DataFrame(columns=['t_start', 't_end', 'pos', 'predict_landcover'], index=np.arange(0, len(cold_block)))
        df_predict_each_rec_cg['t_start'] = array_t_start
        df_predict_each_rec_cg['t_end'] = array_t_end
        df_predict_each_rec_cg['pos'] = array_pos
        df_predict_each_rec_cg['predict_landcover'] = array_LC

        # output the classification results, stored by segments, not image
        output_json = join(output_path, 'json_file')
        if not os.path.exists(output_json):
            os.makedirs(output_json, exist_ok=True)

        output_filename_eachblock = join(output_json, '{}_{}_LC.json'.format(tilename, blockname))
        logging.info('output the classification results for {}'.format(output_filename_eachblock))

        with open(output_filename_eachblock, 'w') as f:
            js.dump(df_predict_each_rec_cg.to_json(), f)



def post_processing(y_prediction, y_prediction_prob, location_pos, location_t_end, location_t_start):
    """
        post-processing the classification results output by random forest.
        Rules included:
        (1) primary forest rule after 1996: correct the primary forest when change is detected
        (2) developed rule: reduce the errors in misclassified developed pixels in the middle, such as shrub -> developed -> shrub
        (3) third rule to process the last temporal segment
    Args:
        y_prediction:
        y_prediction_prob:
        location_pos:
        location_t_end:
        location_t_start:

    Returns:
    """

    for i_loc_pos in range(0, len(location_pos)):
        # (1) primary forest rule since 1996, i.e., if primary forest change was detected after 1996, then convert it to second possible land cover types
        if (y_prediction[location_pos[i_loc_pos]] in [3, 4]) & (i_loc_pos != 0):

            if location_t_end[i_loc_pos - 1] > (datetime_to_datenum(datetime(year=1996, month=1, day=1))):
                prob_list_LC = y_prediction_prob[location_pos[i_loc_pos]]
                position_second_prob = heapq.nlargest(2, np.arange(len(prob_list_LC)),
                                                      key=prob_list_LC.__getitem__)[-1] + 1
                if position_second_prob in [3, 4]:
                    position_second_prob = heapq.nlargest(3, np.arange(len(prob_list_LC)),
                                                          key=prob_list_LC.__getitem__)[-1] + 1

                y_prediction[location_pos[i_loc_pos]] = position_second_prob

    for i_loc_pos in range(len(location_pos) - 2, -1, -1):
        # (2) developed rule
        if (i_loc_pos == len(location_pos) - 2) & (y_prediction[location_pos[i_loc_pos]] == 1):
            # if this is the second-to-last segment, if the last segment is not develop or barren and the segment time span exceeds 3 years,
            # correct the developed pixel
            # The corrected land cover type cannot be primary wet or primary dry forests
            if (y_prediction[location_pos[i_loc_pos + 1]] in [3, 4, 5, 6, 7, 8, 9, 10]) & ((location_t_end[i_loc_pos + 1] - location_t_start[i_loc_pos + 1]) > 3 * 365):
                prob_list_LC = y_prediction_prob[location_pos[i_loc_pos]]
                position_second_prob = heapq.nlargest(2, np.arange(len(prob_list_LC)),
                                                      key=prob_list_LC.__getitem__)[-1] + 1
                if position_second_prob in [3, 4]:
                    position_second_prob = heapq.nlargest(3, np.arange(len(prob_list_LC)),
                                                          key=prob_list_LC.__getitem__)[-1] + 1
                    if position_second_prob in [3, 4]:
                        position_second_prob = heapq.nlargest(4, np.arange(len(prob_list_LC)),
                                                              key=prob_list_LC.__getitem__)[-1] + 1

                y_prediction[location_pos[i_loc_pos]] = position_second_prob
        elif (i_loc_pos == len(location_pos) - 2) & (y_prediction[location_pos[i_loc_pos]] != 1):
            pass
        else:
            if y_prediction[location_pos[i_loc_pos]] == 1:
                # if there are two more segments
                # if one of the next two segments is not developed or barren, correct the developed pixel
                # The corrected land cover type cannot be primary wet or primary dry forests
                if (y_prediction[location_pos[i_loc_pos + 1]] in [2, 3, 4, 5, 6, 7, 8, 9, 10]) & (y_prediction[location_pos[i_loc_pos + 2]] in [2, 3, 4, 5, 6, 7, 8, 9, 10]):
                    prob_list_LC = y_prediction_prob[location_pos[i_loc_pos]]
                    position_second_prob = heapq.nlargest(2, np.arange(len(prob_list_LC)),
                                                          key=prob_list_LC.__getitem__)[-1] + 1
                    if position_second_prob in [3, 4]:
                        position_second_prob = heapq.nlargest(3, np.arange(len(prob_list_LC)),
                                                              key=prob_list_LC.__getitem__)[-1] + 1
                        if position_second_prob in [3, 4]:
                            position_second_prob = heapq.nlargest(4, np.arange(len(prob_list_LC)),
                                                                  key=prob_list_LC.__getitem__)[-1] + 1

                    y_prediction[location_pos[i_loc_pos]] = position_second_prob

    for i_loc_pos in range(len(location_pos) - 1, len(location_pos)):
        # (3) rule for post-processing the last segment, if the last segment is not the change from primary forest and last segment starts from 2022,
        # no land cover change will be made
        # This is because the last segment is too short to get the stable COLD fitting curve for classification
        if (y_prediction[location_pos[i_loc_pos - 1]] in [3, 4]):
            pass
        elif location_t_start[i_loc_pos] >= (datetime_to_datenum(datetime(year=2022, month=1, day=1))):
            y_prediction[location_pos[i_loc_pos]] = y_prediction[location_pos[i_loc_pos - 1]]

    return y_prediction


def img_lcmap_block_acquire(output_path, tilename, blockname):
    """
        generate the land cover map from the output json file
        Args:
            output_path: the root output folder
            tilename
            blockname
        Returns:
            img_output_lcmap: the generated land cover map
    """

    img_output_lcmap = np.zeros((len(np.arange(record_start_year, record_end_year + 1)), Block_Size, Block_Size), dtype=np.float32) + 9999.0

    filename_json = join(output_path, 'json_file', '{}_{}_LC.json'.format(tilename, blockname))
    with open(filename_json) as f:
        data = js.load(f)
    df_predict_each_rec_cg = pd.read_json(data).sort_index()

    array_t_start = df_predict_each_rec_cg['t_start']
    array_t_end = df_predict_each_rec_cg['t_end']
    array_pos = df_predict_each_rec_cg['pos'] - 1
    array_LC = df_predict_each_rec_cg['predict_landcover']

    # row_id_block = int(blockname[3:7])
    # col_id_block = int(blockname[10:14])
    for seg_id in range(0, len(df_predict_each_rec_cg)):
        row_id_image, col_id_image = int(array_pos[seg_id] // Block_Size), array_pos[seg_id] % Block_Size
        # row_id_in_tile_upletf = row_id_block + row_id_image
        # col_id_in_tile = col_id_block + col_id_image

        start_year = datenum_to_datetime(array_t_start[seg_id]).year
        end_year = datenum_to_datetime(array_t_end[seg_id]).year

        if start_year < record_start_year:
            if end_year < record_start_year:
                pass
            elif (end_year >= record_start_year) & (end_year < record_end_year):
                img_output_lcmap[0:end_year - record_start_year + 1, row_id_image, col_id_image] = array_LC[seg_id]
            else:
                img_output_lcmap[:, row_id_image, col_id_image] = array_LC[seg_id]
        elif (start_year >= record_start_year) & (start_year <= record_end_year):
            if end_year >= record_end_year:
                img_output_lcmap[start_year - record_start_year::, row_id_image, col_id_image] = array_LC[seg_id]
            else:
                img_output_lcmap[start_year - record_start_year:end_year - record_start_year + 1, row_id_image, col_id_image] = array_LC[seg_id]
        else:
            pass

    img_output_lcmap[img_output_lcmap == 9999.0] = np.nan

    return img_output_lcmap


def landcover_merge(img_lcmap_fill):
    """
        merge the secondary wet and dry forest to secondary forest
        Ten types land cover:
        1 -> Developed
        2 -> Barren
        3 -> Primary wet forest
        4 -> Primary dry forest
        5 -> Secondary wet forest
        6 -> Secondary dry forest
        7 -> Shrub/Grass
        8 -> Cropland
        9 -> Water
        10 -> Wetland

        After merging, nine land cver types
        1 -> Developed
        2 -> Barren
        3 -> Primary wet forest
        4 -> Primary dry forest
        5 -> Secondary forest
        6 -> Shrub/Grass
        7 -> Cropland
        8 -> Water
        9 -> Wetland
    """

    img_lcmap_merge = img_lcmap_fill.copy()

    img_lcmap_merge[(img_lcmap_merge == 5) | (img_lcmap_merge == 6)] = 5
    img_lcmap_merge[img_lcmap_merge == 7] = 6
    img_lcmap_merge[img_lcmap_merge == 8] = 7
    img_lcmap_merge[img_lcmap_merge == 9] = 8
    img_lcmap_merge[img_lcmap_merge == 10] = 9

    return img_lcmap_merge


def landcover_fill(img_lcmap_output):
    """
        fill the land cover gap in the time series to ensure the consistent land cover map from 1984 to 2022,
        first using the bfill, then using the ffill
        Args:
            img_lcmap_output: land cover map classified from the COLD rec_cg file
        Returns:
            img_lcmap_fill: the seamless land cover map
    """

    MIN_LC = int(np.nanmin(img_lcmap_output))
    MAX_LC = int(np.nanmax(img_lcmap_output))

    img_lcmap_fill = img_lcmap_output.copy()

    col_count = np.shape(img_lcmap_fill)[-1]

    masknan = (img_lcmap_fill > MAX_LC) | (img_lcmap_fill < MIN_LC)
    img_lcmap_fill[masknan] = np.nan

    for col_id in range(0, col_count):
        df_temp = pd.DataFrame(img_lcmap_fill[:, :, col_id])
        if np.isnan(df_temp).all().all():
            pass
        else:
            if np.isnan(df_temp).any().any():
                img_lcmap_fill[:, :, col_id] = df_temp.fillna(method='bfill')
                img_lcmap_fill[:, :, col_id] = df_temp.fillna(method='ffill')

    return img_lcmap_fill


def landcover_output(img_lcmap_fill, tilename, blockname, src_geotrans, src_proj, output_path):
    """
        output the land cover map
        Args:
            img_lcmap_fill: the seamless land cover map
            tilename: tile name
            src_proj: the proj info
            src_geotrans: geolocation info
            output_path: the folder to save the classification results

        Returns:
            img_lcmap_fill: the seamless land cover map
    """

    src_geotrans = (src_geotrans[0] + int(blockname[-4:]) * src_geotrans[1],
                    src_geotrans[1],
                    src_geotrans[2],
                    src_geotrans[3] + int(blockname[3: 7]) * src_geotrans[5],
                    src_geotrans[4],
                    src_geotrans[5])

    output_path_lcmap = os.path.join(output_path, 'LC_eachyear_{}_{}'.format(record_start_year, record_end_year))
    if not os.path.exists(output_path_lcmap):
        os.makedirs(output_path_lcmap, exist_ok=True)

    output_filename = os.path.join(output_path_lcmap, '{}_{}_lcmap.tif'.format(tilename, blockname))

    tif_output = gdal.GetDriverByName('GTiff').Create(output_filename, Block_Size, Block_Size,
                                                       record_end_year - record_start_year + 1, gdalconst.GDT_Float32)
    tif_output.SetGeoTransform(src_geotrans)
    tif_output.SetProjection(src_proj)

    for year_id in range(0, record_end_year - record_start_year + 1):
        img_output_eachyear = img_lcmap_fill[year_id, :, :].copy()
        Band = tif_output.GetRasterBand(year_id + 1)
        Band.WriteArray(img_output_eachyear)

    tif_output = None


def land_cover_plot(img_plot, title='', figsize=(16, 9.5), landcover_system=None, colors=None, ticks_flag=False):
    """
        plot the land cover
    Args:
        img_plot:
        title:
        figsize:
        landcover_system: dictionary to record the land cover system
        colors: colors for each land cover type
        ticks_flag: flag to determine whether to show the ticks, default is False
    Returns:

    """
    sns.set_style("white")
    if landcover_system is None:
        landcover_system = {'1': 'Developed',
                            '2': 'Barren',
                            '3': 'PrimaryWetForest',
                            '4': 'PrimaryDryForest',
                            '5': 'SecondaryForest',
                            '6': 'ShrubGrass',
                            '7': 'Cropland',
                            '8': 'Water',
                            '9': 'Wetland'}

        colors = np.array([np.array([241, 1, 0, 255]) / 255,
                           np.array([179, 175, 164, 255]) / 255,
                           np.array([29, 101, 51, 255]) / 255,
                           np.array([244, 127, 17, 255]) / 255,
                           np.array([108, 169, 102, 255]) / 255,
                           np.array([208, 209, 129, 255]) / 255,
                           np.array([174, 114, 41, 255]) / 255,
                           np.array([72, 109, 162, 255]) / 255,
                           np.array([200, 230, 248, 255]) / 255
                           ])

    vmin, vmax = 1, len(landcover_system)

    img = img_plot.copy()
    img = img.astype(float)
    img[img < vmin] = np.nan
    img[img > vmax] = np.nan

    unique_landcover = np.unique(img)[~np.isnan(np.unique(img))]
    unique_landcover = (unique_landcover).astype(int)

    title_legend = []
    mask_color = np.zeros(len(colors), dtype=bool)
    for p in unique_landcover:
        mask_color[p - 1] = True
        title_legend.append('{} {}'.format(p, landcover_system[str(p)]))
    colors_plot = colors[mask_color]

    labelsize = 20
    title_label_size = 24
    ticklength = 6
    axes_linewidth = 1.5
    legend_size = 16

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)

    matplotlib.rcParams['axes.linewidth'] = axes_linewidth
    for i in axes.spines.values():
        i.set_linewidth(axes_linewidth)

    cmap = matplotlib.colors.ListedColormap(colors)
    boundaries = np.arange(1, len(landcover_system) + 2)
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    im = axes.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')

    axes.tick_params('x', labelsize=labelsize, direction='out', length=ticklength, width=axes_linewidth, bottom=ticks_flag, labelbottom=ticks_flag, which='major')
    axes.tick_params('y', labelsize=labelsize, direction='out', length=ticklength, width=axes_linewidth, left=ticks_flag, labelleft=ticks_flag, which='major')

    patches = [mpatches.Patch(color=colors_plot[i], label=title_legend[i]) for i in range(0, len(colors_plot))]
    plt.legend(handles=patches, bbox_to_anchor=(1, 0.92, 0.1, 0.1), loc='upper left', borderaxespad=0.5, fontsize=legend_size)

    plt.title(title, fontsize=title_label_size, pad=1.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    post_processing_flag = 0
    landcover_version = 'v2'
    tilename = 'h05v02'

    # list_blockname = ['row1000col0000', 'row1000col0250', 'row1000col0500',
    #                   'row1250col0000', 'row1250col0250', 'row1250col0500']
    # for blockname in list_blockname:

    blockname = 'row1000col0000'

    if post_processing_flag == 0:
        post_processing_des = 'No rule applied'
    elif post_processing_flag == 1:
        post_processing_des = 'PF rule after 1996 & Develop correction 1996 and 2022 & last segment rule'
    else:
        post_processing_des = None

    output_rootpath = get_output_rootpath(landcover_version)  # get the root output directory
    output_path = get_output_path(landcover_version, tilename)  # get the output path for the tile

    src_geotrans, src_proj = get_projection_info(tilename)

    logging.basicConfig(filename=join(output_rootpath, '{}.log'.format(landcover_version)),
                        level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    logging.info('calibration version {}'.format(landcover_version))
    logging.info('post processing rule: {} {}'.format(post_processing_flag, post_processing_des))
    logging.info('land cover output version {}'.format(landcover_version))

    # read pre-trained random forest classifier
    output_name_rf = join(rootpath, 'data', 'random_forest_model', 'rf_classifier_i01.joblib')
    rf_classifier = joblib.load(output_name_rf)

    img_dem_block, img_slope_block, img_aspect_block = read_dem(tilename, blockname)  # read topography information

    filename_cold_results = join(rootpath, 'data', 'cold_reccg', tilename, '{}_reccg.npy'.format(blockname))
    cold_block = read_cold_reccg(filename_cold_results)   # read COLD coefficients

    # land cover classification using random forest, post-processing is applied
    land_cover_classification(rf_classifier, cold_block, img_dem_block, img_slope_block, img_aspect_block, post_processing_flag,
                              output_path, tilename, blockname)

    # get the land cover images from the temporal segments
    img_lcmap_eachblock = img_lcmap_block_acquire(output_path, tilename, blockname)

    # fill the missing values
    img_lcmap_fill = landcover_fill(img_lcmap_eachblock)

    # merge secondary wet and secondary dry forests to secondary forest
    img_lcmap_merge = landcover_merge(img_lcmap_fill)

    # output the land cover map
    landcover_output(img_lcmap_merge, tilename, blockname, src_geotrans, src_proj, output_path)

    # plot the land cover
    land_cover_plot(img_lcmap_merge[-1], title='land cover in 2022')

