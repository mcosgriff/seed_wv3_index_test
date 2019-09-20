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
        band_1_values = get_band_pixel_values(raster, BandTable.COASTAL)
        band_6_values = get_band_pixel_values(raster, BandTable.RED_EDGE)

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
        band_3_values = get_band_pixel_values(raster, BandTable.GREEN)
        band_4_values = get_band_pixel_values(raster, BandTable.YELLOW)

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
        band_10_values = get_band_pixel_values(raster, BandTable.SWIR_2)
        band_12_values = get_band_pixel_values(raster, BandTable.SWIR_4)
        band_13_values = get_band_pixel_values(raster, BandTable.SWIR_5)
        band_16_values = get_band_pixel_values(raster, BandTable.SWIR_8)

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
        band_10_values = get_band_pixel_values(raster, BandTable.SWIR_2)
        band_11_values = get_band_pixel_values(raster, BandTable.SWIR_3)
        band_14_values = get_band_pixel_values(raster, BandTable.SWIR_6)
        band_16_values = get_band_pixel_values(raster, BandTable.SWIR_8)

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
        band_1_values = get_band_pixel_values(raster, BandTable.COASTAL)
        band_8_values = get_band_pixel_values(raster, BandTable.NEAR_INFRARED_2)

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(band_1_values > 0, band_8_values > 0)
        output_ds = np.where(check, (band_8_values - band_1_values) / (band_8_values + band_1_values), -9)

        return save_output(wv3_file, 'wv_water_index', build_output_profile(raster), output_ds)


def save_output(wv3_file: str, postfix: str, profile: property, output_ds: np.ndarray) -> str:
    with rasterio.open(os.path.splitext(wv3_file)[0] + '_{}.tif'.format(postfix), mode='w', **profile) as output:
        output.write(output_ds, 1)

        return output.name


def normalized_differential_vegetation_index(wv3_file: str) -> str:
    """

    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        red_values = get_band_pixel_values(raster, BandTable.RED)
        nir_values = get_band_pixel_values(raster, BandTable.NEAR_INFRARED_1)

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(red_values > 0, nir_values > 0)
        output_ds = np.where(check, (nir_values - red_values) / (nir_values + red_values), -9)

        return save_output(wv3_file, 'ndvi', build_output_profile(raster), output_ds)


def get_band_pixel_values(raster: rasterio.DatasetReader, which_band: BandTable,
                          array_type: str = 'float32') -> np.ndarray:
    band = raster.read(which_band.value)
    return band.astype(array_type)


def build_output_profile(raster: rasterio.DatasetReader) -> property:
    profile = raster.profile
    profile.update(count=1, dtype=rasterio.float32, compress=Compression.deflate.value)

    return profile


def process_image(which_algorithm: Index, wv3_file: str) -> str:
    return eval(which_algorithm.value.format(wv3_file))
