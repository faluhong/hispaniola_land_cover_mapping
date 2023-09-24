"""
generate the singele path .tif file for running
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

pwd = os.getcwd()
rootpath = os.path.abspath(os.path.join(pwd, '..'))
path_pythoncode = os.path.join(rootpath, 'pythoncode')
sys.path.append(path_pythoncode)

if __name__ == '__main__':

    RES = 30
    dst_proj = 'PROJCS["Albers_Conic_Equal_Area",GEOGCS["WGS 84",DATUM["WGS_1984",' \
               'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
               'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],' \
               'PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],' \
               'PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],' \
               'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

    filename_global_singlepath = join(rootpath, 'data', 'single_path', 'NonoverlapLandsatPathsGlobal.tif')
    obj = gdal.Open(filename_global_singlepath)

    # project the original file into the VRT format, determine the boundary of the image
    params_VRT = gdal.WarpOptions(format='VRT', dstSRS=dst_proj, xRes=RES, yRes=RES,
                                  resampleAlg=gdal.GRIORA_NearestNeighbour, dstNodata=255)
    dst_boundary = gdal.Warp(destNameOrDestDS='', srcDSOrSrcDSTab=obj, options=params_VRT)

    dstboundary_geotrans = dst_boundary.GetGeoTransform()
    XSize, YSize = dst_boundary.RasterXSize, dst_boundary.RasterYSize

    xLeft = dstboundary_geotrans[0]
    yTop = dstboundary_geotrans[3]

    xRight = xLeft + XSize * RES
    yBottom = yTop - YSize * RES

    p_img = Polygon([(xLeft, yBottom), (xLeft, yTop), (xRight, yTop), (xRight, yBottom)])   # the polygon of the LandsatPathGlobal tif file

    path_shp = join(rootpath, 'data', 'tile_grid', 'individual_grid')
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
                params = gdal.WarpOptions(format='GTiff', dstSRS=dst_proj,
                                          outputBounds=[xmin_shp, ymin_shp, xmax_shp, ymax_shp], xRes=RES, yRes=RES,
                                          resampleAlg=gdal.GRIORA_NearestNeighbour, dstNodata=255)

                dst_filename = join(rootpath, 'data', 'single_path', '{}.tif'.format(tilename))

                # clip the NonoverlapLandsatPathsGlobal.tif to each tile
                dst = gdal.Warp(destNameOrDestDS=dst_filename, srcDSOrSrcDSTab=obj, options=params)
                img_single_path = dst.ReadAsArray()
                dst = None
                del dst

                print(tilename, np.unique(img_single_path))

            except:
                pass
