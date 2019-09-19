import os
from enum import Enum

import numpy as np
import rasterio
from rasterio.enums import Compression


class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


class Index(NoValue):
    WORLD_VIEW_WATER_INDEX = 'world_view_water_index("{}")'


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


def world_view_water_index(wv3_file: str) -> np.ndarray:
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        red = raster.read(BandTable.RED.value)
        nir = raster.read(BandTable.NEAR_INFRARED_1.value)

        red_values = red.astype('float32')
        nir_values = nir.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        ndvi = np.empty(raster.shape, dtype=rasterio.float32)
        check = np.logical_or(red_values > 0, nir_values > 0)
        ndvi = np.where(check, (nir_values - red_values) / (nir_values + red_values), -9)

        return ndvi


def get_source_profile(wv3_file: str) -> property:
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        return raster.profile


def process_image(which_algorithm: Index, wv3_file: str) -> str:
    dst_ds = eval(which_algorithm.value.format(wv3_file))

    profile = get_source_profile(wv3_file)
    profile.update(count=1, dtype=rasterio.float32, compress=Compression.deflate.value)

    with rasterio.open(os.path.splitext(wv3_file)[0] + '_ndvi.tif', mode='w', **profile) as ndvi_file:
        ndvi_file.write(dst_ds, 1)

        return ndvi_file.name
