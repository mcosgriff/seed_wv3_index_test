import os
from enum import Enum

import numpy as np
import rasterio
from rasterio.enums import Compression


class BandTable(Enum):
    COASTAL = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4
    RED = 5
    RED_EDGE = 6
    NEAR_INFRARED_1 = 7
    NEAR_INFRARED_2 = 8
    SWIR_1 = 9
    SWIR_2 = 10
    SWIR_3 = 11
    SWIR_4 = 12
    SWIR_5 = 13
    SWIR_6 = 14
    SWIR_7 = 15
    SWIR_8 = 16


def run() -> None:
    filename = '/Users/mcosgriff/Downloads/18FEB23084801-A3DS-057798936010_cal_2016v0_scube_native.tif'
    with rasterio.open(filename, mode='r', driver='GTiff') as raster:
        red = raster.read(BandTable.RED.value)
        nir = raster.read(BandTable.NEAR_INFRARED_1.value)

        red_values = red.astype('float32')
        nir_values = nir.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        ndvi = np.empty(raster.shape, dtype=rasterio.float32)
        check = np.logical_or(red_values > 0, nir_values > 0)
        ndvi = np.where(check, (nir_values - red_values) / (nir_values + red_values), -9)

        profile = raster.profile
        profile.update(count=1, dtype=rasterio.float32, compress=Compression.deflate.value)

        with rasterio.open(os.path.splitext(filename)[0] + '_ndvi.tif', mode='w', **profile) as ndvi_file:
            ndvi_file.write(ndvi, 1)

        '''for x, y in it.product(range(raster.width), range(raster.height)):
            red_pix = red[x][y]
            nir_pix = nir[x][y]'''


if __name__ == '__main__':
    run()
