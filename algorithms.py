import logging
import math
import os
from enum import Enum

import numpy as np
import rasterio
from rasterio.enums import Compression


class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


class Index(NoValue):
    NDVI = 'normalized_differential_vegetation_index("{}")'
    WORLD_VIEW_WATER = 'world_view_water_index("{}")'
    POLYMER_1 = 'polymer_1_index("{}")'
    POLYMER_2 = 'polymer_2_index("{}")'
    SOIL = 'soil_index("{}")'
    BUILT_UP = 'built_up_index("{}")'
    NDVI_RE = 'normalized_differential_vegetation_red_edge_index("{}")'


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
        band_1 = get_band_pixel_values(raster, BandTable.COASTAL)
        band_6 = get_band_pixel_values(raster, BandTable.RED_EDGE)

        np.seterr(divide='ignore', invalid='ignore')
        ds_temp_1 = np.subtract(band_1, band_6)
        ds_temp_2 = np.add(band_1, band_6)

        output_ds = np.where(ds_temp_2 > 0, np.divide(ds_temp_1, ds_temp_2), float('nan'))

        return save_output(wv3_file, 'built_up_index', build_output_profile(raster), output_ds)


def soil_index(wv3_file: str) -> str:
    """
    Detect and differentitate exposed soil
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_3 = get_band_pixel_values(raster, BandTable.GREEN)
        band_4 = get_band_pixel_values(raster, BandTable.YELLOW)

        np.seterr(divide='ignore', invalid='ignore')

        output_ds = np.divide(np.subtract(band_4, band_3), np.add(band_4, band_3))

        return save_output(wv3_file, 'soil_index', build_output_profile(raster), linear_percent_stretch(output_ds))


def polymer_1_index(wv3_file: str) -> str:
    """
    Detect certain types of polymer chemistry
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_10 = get_band_pixel_values(raster, BandTable.SWIR_2)
        band_12 = get_band_pixel_values(raster, BandTable.SWIR_4)
        band_13 = get_band_pixel_values(raster, BandTable.SWIR_5)
        band_16 = get_band_pixel_values(raster, BandTable.SWIR_8)

        np.seterr(divide='ignore', invalid='ignore')

        temp_ds_1 = np.where(band_12 > 0, np.divide(band_10, band_12), np.nan)
        temp_ds_2 = np.where(band_16 > 0, np.divide(band_13, band_16), np.nan)

        check = np.logical_or(temp_ds_1 != np.nan, temp_ds_2 != np.nan)

        output_ds = np.where(check, temp_ds_1 + temp_ds_2, np.nan)

        return save_output(wv3_file, 'polymer_1_index', build_output_profile(raster), output_ds)


def polymer_2_index(wv3_file: str) -> str:
    """
    Detect certain types of polymer chemistry
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_10 = get_band_pixel_values(raster, BandTable.SWIR_2)
        band_11 = get_band_pixel_values(raster, BandTable.SWIR_3)
        band_14 = get_band_pixel_values(raster, BandTable.SWIR_6)
        band_16 = get_band_pixel_values(raster, BandTable.SWIR_8)

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(np.logical_or(band_10 > 0, band_11 > 0),
                              np.logical_or(band_14 > 0, band_16 > 0))
        output_ds = np.where(check, ((band_10 / band_11) + (band_14 / band_16)), -9)

        return save_output(wv3_file, 'polymer_2_index', build_output_profile(raster), output_ds)


def world_view_water_index(wv3_file: str) -> str:
    """
    Detect standing, flowing, or absorbed water
    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_1 = get_band_pixel_values(raster, BandTable.COASTAL)
        band_8 = get_band_pixel_values(raster, BandTable.NEAR_INFRARED_2)

        np.seterr(divide='ignore', invalid='ignore')
        check = np.logical_or(band_1 > 0, band_8 > 0)
        output_ds = np.where(check, ((band_1 - band_8) / (band_1 + band_8)), -9)

        return save_output(wv3_file, 'wv_water_index', build_output_profile(raster), output_ds)


def save_output(wv3_file: str, postfix: str, profile: property, output_ds: np.ndarray) -> str:
    with rasterio.open(os.path.splitext(wv3_file)[0] + '_{}.tif'.format(postfix), mode='w', **profile) as output:
        output.write(output_ds, 1)

        return output.name


def linear_percent_stretch(image: np.ndarray, percent=2) -> np.ndarray:
    """
    A linear percent stretch allows you to trim extreme values from both ends of the histogram using a specified
    percentage.
    :param image:
    :param percent:
    :return:
    """
    std_dev = np.nanstd(image)
    mean = np.nanmean(image)

    logging.getLogger('wv3_index_processing').info(
        'Before liner percent stretch: stddev={}, mean={}, min={}, max={}'.format(std_dev, mean, np.nanmin(image),
                                                                                  np.nanmax(image)))

    low_cutoff = std_dev - (math.fabs(mean) * percent)
    high_cutoff = std_dev + (math.fabs(mean) * percent)

    image[image > high_cutoff] = np.nan
    image[image < low_cutoff] = np.nan

    logging.getLogger('wv3_index_processing').info(
        'After liner percent stretch: stddev={}, mean={}, min={}, max={}'.format(np.nanstd(image), np.nanmean(image),
                                                                                 np.nanmin(image),
                                                                                 np.nanmax(image)))

    return image


def image_histogram_equalization(image: np.ndarray, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # from https://stackoverflow.com/a/28520445

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True,
                                         range=(np.nanmin(image), np.nanmax(image)))
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize , [-1] returns only the last element in the array

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)  # [:-1] returns all elements except the last one

    return image_equalized.reshape(image.shape).astype("float32"), cdf


def normalized_differential_vegetation_index(wv3_file: str) -> str:
    """

    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_5 = get_band_pixel_values(raster, BandTable.RED)
        band_7 = get_band_pixel_values(raster, BandTable.NEAR_INFRARED_1)

        np.seterr(divide='ignore', invalid='ignore')
        temp_ds_1 = np.subtract(band_7, band_5)
        temp_ds_2 = np.add(band_7, band_5)

        check = np.logical_or(temp_ds_1 > 0, temp_ds_2 > 0)
        output_ds = np.where(temp_ds_2 > 0, np.divide(temp_ds_1, temp_ds_2), math.nan)

        return save_output(wv3_file, 'ndvi', build_output_profile(raster), output_ds)


def normalized_differential_vegetation_red_edge_index(wv3_file: str) -> str:
    """

    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_5 = get_band_pixel_values(raster, BandTable.RED)
        band_6 = get_band_pixel_values(raster, BandTable.RED_EDGE)

        np.seterr(divide='ignore', invalid='ignore')
        temp_ds_1 = np.subtract(band_6, band_5)
        temp_ds_2 = np.add(band_6, band_5)

        check = np.logical_or(temp_ds_1 > 0, temp_ds_2 > 0)
        output_ds = np.where(temp_ds_2 > 0, np.divide(temp_ds_1, temp_ds_2), math.nan)

        return save_output(wv3_file, 'ndvi_re', build_output_profile(raster), output_ds)


def get_band_pixel_values(raster: rasterio.DatasetReader, which_band: BandTable,
                          array_type: str = 'float32') -> np.ndarray:
    band = raster.read(which_band.value)
    return band.astype(array_type)


def build_output_profile(raster: rasterio.DatasetReader) -> property:
    profile = raster.profile
    profile.update(count=1, dtype=rasterio.float32, compress=Compression.packbits.value)

    return profile


def process_image(which_algorithm: Index, wv3_file: str) -> str:
    logging.getLogger('wv3_index_processing').info('Running {} on {}'.format(which_algorithm.name, wv3_file))
    return eval(which_algorithm.value.format(wv3_file))
