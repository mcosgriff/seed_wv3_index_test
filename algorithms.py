import logging
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
        first = np.subtract(band_1, band_6)
        second = np.add(band_1, band_6)

        check = np.logical_or(first > 0, band_6 > 0)
        # output_ds = np.where(check, (band_6_values - band_1_values) / (band_6_values + band_1_values), -9)
        output_ds = np.where(check, np.divide(first, second), -9)

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
        check = np.logical_or(band_3 > 0, band_4 > 0)
        output_ds = np.where(check, (band_4 - band_3) / (band_4 + band_3), -9)

        return save_output(wv3_file, 'soil_index', build_output_profile(raster), output_ds)


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
        # check = np.logical_or(np.logical_or(band_10_values > 0, band_12_values > 0),
        # np.logical_or(band_13_values > 0, band_16_values > 0))
        # check = np.logical_and(band_12_values > 0, band_16_values > 0)
        # output_ds = np.where(check, (band_10_values / band_12_values) + (band_13_values / band_16_values), float(-9))
        # output_ds = ((band_10_values / band_12_values) + (band_13_values / band_16_values))

        # masked_output_ds = np.ma.masked_equal(output_ds, float(-9))

        check_1 = np.logical_or(band_10 > 0, band_12 > 0)
        check_2 = np.logical_or(band_13 > 0, band_16 > 0)

        temp_ds_1 = np.where(check_1, band_10 / band_12, -9)
        temp_ds_2 = np.where(check_2, band_13 / band_16, -9)

        check_3 = np.logical_or(temp_ds_1 == -9, temp_ds_2 == -9)

        output_ds = np.where(check_3, temp_ds_1 + temp_ds_2, -9)

        # unique, counts = np.unique(output_ds, return_counts=True)
        # unique_counts = dict(zip(unique, counts))

        # logging.debug('Unique counts = {}'.format(unique_counts))
        # hist, bin_edges = np.histogram(output_ds)
        # width = 0.7 * (bin_edges[1] - bin_edges[0])
        # center = (bin_edges[:-1] + bin_edges[1:]) / 2

        # fig, ax = plt.subplots()
        # ax.bar(bin_edges, hist, width=np.diff(bin_edges), ec="k", align='edge')

        # plt.hist(x=hist, bins=bin_edges[:-1], density=False, histtype='bar', color='b', edgecolor='k', alpha=0.5)
        # plt.line(center, hist, align='center', width=width)
        # plt.hist(bin_edges, bins=50)
        # max_freq = hist.max()
        # plt.ylim(ymax=np.ceil(max_freq/10) * 10 if max_freq % 10 else max_freq + 10)
        # plt.xlabel('Pixel Value')
        # plt.xticks(bin_edges[:-1])
        # plt.ylabel('Occurances')
        # plt.title('Pixel Count for Index')

        # plt.show()

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


def normalized_differential_vegetation_index(wv3_file: str) -> str:
    """

    :param wv3_file: WV3 16 band image
    :return: Processed filename
    """
    with rasterio.open(wv3_file, mode='r', driver='GTiff') as raster:
        band_5 = get_band_pixel_values(raster, BandTable.RED)
        band_7 = get_band_pixel_values(raster, BandTable.NEAR_INFRARED_1)

        np.seterr(divide='ignore', invalid='ignore')
        first = np.subtract(band_7, band_5)
        second = np.add(band_7, band_5)

        check = np.logical_or(first > 0, second > 0)
        output_ds = np.where(check, np.divide(first, second), -9)

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
        first = np.subtract(band_6, band_5)
        second = np.add(band_6, band_5)

        check = np.logical_or(first > 0, second > 0)
        output_ds = np.where(check, np.divide(first, second), -9)

        return save_output(wv3_file, 'ndvi_re', build_output_profile(raster), output_ds)


def get_band_pixel_values(raster: rasterio.DatasetReader, which_band: BandTable,
                          array_type: str = 'float32') -> np.ndarray:
    band = raster.read(which_band.value)
    return band.astype(array_type)


def build_output_profile(raster: rasterio.DatasetReader) -> property:
    profile = raster.profile
    profile.update(count=1, dtype=rasterio.float32, compress=Compression.deflate.value)

    return profile


def add_ndarray_values(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.add(first, second)


def divide_ndarray_values(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    np.divide(first, second)


def process_image(which_algorithm: Index, wv3_file: str, _logger: logging.Logger) -> str:
    logger = _logger

    logging.info('Runing {} on {}'.format(which_algorithm.name, wv3_file))
    return eval(which_algorithm.value.format(wv3_file))
