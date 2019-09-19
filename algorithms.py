import os
from enum import Enum

import numpy as np
import rasterio
from rasterio.enums import Compression


class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


class Index(NoValue):
    NORMALIZED_DIFFERENTIAL_VEGETATION = 'normalized_differential_vegetation_index("{}")'
    WORLD_VIEW_WATER = 'world_view_water_index("{}")'
    POLYMER_1 = 'polymer_1_index("{}")'
    POLYMER_2 = 'polymer_2_index("{}")'
    SOIL = 'soil_index("{}")'
    BUILT_UP = 'built_up_index("{}")'


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


def built_up_index(wv3_file: str) -> str:
    """
    Detect surfaces such as building and roads
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_1 = raster.read(BandTable.COASTAL.value)
        band_6 = raster.read(BandTable.RED_EDGE.value)

        band_1_values = band_1.astype('float32')
        band_6_values = band_6.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(band_1_values > 0, band_6_values > 0)
        output_ds = np.where(check, (band_6_values - band_1_values) / (band_6_values + band_1_values), -9)

        return save_output(wv3_file, 'built_up_index', build_output_profile(raster), output_ds)


def soil_index(wv3_file: str) -> str:
    """
    Detect and differentitate exposed soil
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_3 = raster.read(BandTable.GREEN.value)
        band_4 = raster.read(BandTable.YELLOW.value)

        band_3_values = band_3.astype('float32')
        band_4_values = band_4.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(band_3_values > 0, band_4_values > 0)
        output_ds = np.where(check, (band_4_values - band_3_values) / (band_4_values + band_3_values), -9)

        return save_output(wv3_file, 'soil_index', build_output_profile(raster), output_ds)


def polymer_1_index(wv3_file: str) -> str:
    """
    Detect certain types of polymer chemistry
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_10 = raster.read(BandTable.SWIR_2.value)
        band_12 = raster.read(BandTable.SWIR_4.value)
        band_13 = raster.read(BandTable.SWIR_2.value)
        band_16 = raster.read(BandTable.SWIR_4.value)

        band_10_values = band_10.astype('float32')
        band_12_values = band_12.astype('float32')
        band_13_values = band_13.astype('float32')
        band_16_values = band_16.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(np.logical_or(band_10_values > 0, band_12_values > 0),
                              np.logical_or(band_13_values > 0, band_16_values > 0))
        output_ds = np.where(check, (band_10_values / band_12_values) + (band_13_values / band_16_values), -9)

        return save_output(wv3_file, 'polymer_1_index', build_output_profile(raster), output_ds)


def polymer_2_index(wv3_file: str) -> str:
    """
    Detect certain types of polymer chemistry
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_10 = raster.read(BandTable.SWIR_2.value)
        band_11 = raster.read(BandTable.SWIR_3.value)
        band_14 = raster.read(BandTable.SWIR_6.value)
        band_16 = raster.read(BandTable.SWIR_8.value)

        band_10_values = band_10.astype('float32')
        band_11_values = band_11.astype('float32')
        band_14_values = band_14.astype('float32')
        band_16_values = band_16.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(np.logical_or(band_10_values > 0, band_11_values > 0),
                              np.logical_or(band_14_values > 0, band_16_values > 0))
        output_ds = np.where(check, (band_10_values / band_11_values) + (band_14_values / band_16_values), -9)

        return save_output(wv3_file, 'polymer_2_index', build_output_profile(raster), output_ds)


def world_view_water_index(wv3_file: str) -> str:
    """
    Detect standing, flowing, or absorbed water
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        coastal = raster.read(BandTable.COASTAL.value)
        nir = raster.read(BandTable.NEAR_INFRARED_2.value)

        coastal_values = coastal.astype('float32')
        nir_values = nir.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(coastal_values > 0, nir_values > 0)
        output_ds = np.where(check, (nir_values - coastal_values) / (nir_values + coastal_values), -9)

        return save_output(wv3_file, 'wv_water_index', build_output_profile(raster), output_ds)


def save_output(wv3_file: str, postfix: str, profile: property, output_ds: np.ndarray) -> str:
    with rasterio.open(os.path.splitext(wv3_file)[0] + '_{}_.tif'.format(postfix), mode='w', **profile) as output:
        output.write(output_ds, 1)

        return output.name


def normalized_differential_vegetation_index(wv3_file: str) -> str:
    """

    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        red = raster.read(BandTable.RED.value)
        nir = raster.read(BandTable.NEAR_INFRARED_1.value)

        red_values = red.astype('float32')
        nir_values = nir.astype('float32')

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(red_values > 0, nir_values > 0)
        output_ds = np.where(check, (nir_values - red_values) / (nir_values + red_values), -9)

        return save_output(wv3_file, 'ndvi', build_output_profile(raster), output_ds)


def build_output_profile(raster: rasterio.DatasetReader) -> property:
    profile = raster.profile
    profile.update(count=1, dtype=rasterio.float32, compress=Compression.deflate.value)

    return profile


def process_image(which_algorithm: Index, wv3_file: str) -> str:
    return eval(which_algorithm.value.format(wv3_file))
