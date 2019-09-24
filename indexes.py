import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Compression
from scipy.stats import norm
import logging

from enums import BandTable, StretchTypes
from stretch_types import linear_percent_stretch, image_histogram_equalization


class Indexes:
    def __init__(self, raster_path: str, index_name: str, stretch_type=StretchTypes.NONE):
        self.index_name = index_name
        self.raster_path = raster_path
        self.processed_raster = None
        self.stretch_type = stretch_type
        self.logger = logging.getLogger('wv3_index_processing')

        np.seterr(divide='ignore', invalid='ignore')

    def __enter__(self):
        self.raster = rasterio.open(self.raster_path, mode='r', driver='GTiff')

    def __exit__(self, exc_type, exc_value, traceback):
        self.raster.close()

    def get_band_pixel_values(self, which_band: BandTable, array_type: str = 'float32') -> np.ndarray:
        band = self.raster.read(which_band.value)
        return band.astype(array_type)

    def which_bands_required(self) -> list:
        pass

    def process_index(self) -> (np.ndarray, np.ndarray):
        pass

    def build_output_profile(self) -> property:
        profile = self.raster.profile
        profile.update(count=1, dtype=rasterio.float32, compress=Compression.packbits.value)

        return profile

    def save_output(self, raster: np.ndarray = np.array([])) -> str:
        if raster.size <= 0:
            raster = self.processed_raster

        if raster.size > 0:
            with rasterio.open(os.path.splitext(self.raster_path)[0] + '_{}.tif'.format(self.index_name), mode='w',
                               **self.build_output_profile()) as output:
                output.write(self.processed_raster, 1)

                return output.name

    def index_name(self):
        return self.index_name

    def perform_stretch(self):
        if self.stretch_type == StretchTypes.LINEAR_PERCENT_STRETCH:
            return linear_percent_stretch(self.processed_raster)
        elif self.stretch_type == StretchTypes.IMAGE_HISTOGRAM_EQUALIZATION:
            return image_histogram_equalization(self.processed_raster)
        else:
            return self.processed_raster

    def build_histogram(self, raster: np.ndarray = np.array([])):
        if raster.size > 0:
            raster = self.processed_raster

        if raster.size > 0:
            self.logger.info('Raster={}'.format(raster.shape))

            cleaned_up_raster = np.ravel(raster)
            self.logger.info('Raveled Raster={}'.format(cleaned_up_raster.shape))

            cleaned_up_raster = cleaned_up_raster[~np.isnan(cleaned_up_raster)]
            self.logger.info('Raster with NaN removed={}'.format(cleaned_up_raster.shape))

            mu = np.mean(cleaned_up_raster)
            sigma = np.std(cleaned_up_raster)

            # num_bins = int(math.ceil(np.nanmax(raster) - np.nanmin(raster)))
            num_bins = 30

            fig, ax = plt.subplots()

            # The histogram of the data
            n, bins, patches = ax.hist(cleaned_up_raster, bins=50, density=False, histtype='step')

            # Add a 'best fit' line
            y = norm.pdf(bins, mu, sigma)
            ax.plot(bins, y, '--')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Raster Histogram')

            # Tweak spacing to prevent clipping of y label
            fig.tight_layout()
            plt.show()


class Polymer1(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'polymer_1_index', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.SWIR_2, BandTable.SWIR_4, BandTable.SWIR_5, BandTable.SWIR_8]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_10 = self.get_band_pixel_values(BandTable.SWIR_2)
        band_12 = self.get_band_pixel_values(BandTable.SWIR_4)
        band_13 = self.get_band_pixel_values(BandTable.SWIR_5)
        band_16 = self.get_band_pixel_values(BandTable.SWIR_8)

        np.seterr(divide='ignore', invalid='ignore')

        temp_ds_1 = np.where(band_12 > 0, np.divide(band_10, band_12), np.nan)
        temp_ds_2 = np.where(band_16 > 0, np.divide(band_13, band_16), np.nan)

        check = np.logical_or(temp_ds_1 != np.nan, temp_ds_2 != np.nan)

        self.processed_raster = np.where(check, temp_ds_1 + temp_ds_2, np.nan)

        return self.processed_raster, self.processed_raster, self.perform_stretch()


class Polymer2(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'polymer_2_index', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.SWIR_2, BandTable.SWIR_3, BandTable.SWIR_6, BandTable.SWIR_8]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_10 = self.get_band_pixel_values(BandTable.SWIR_2)
        band_11 = self.get_band_pixel_values(BandTable.SWIR_3)
        band_14 = self.get_band_pixel_values(BandTable.SWIR_6)
        band_16 = self.get_band_pixel_values(BandTable.SWIR_8)

        self.processed_raster = np.add(np.divide(band_10, band_11), np.divide(band_14, band_16))

        return self.processed_raster, self.perform_stretch()


class Soil(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'soil_index', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.GREEN, BandTable.YELLOW]

    def process_index(self) -> np.ndarray:
        band_3 = self.get_band_pixel_values(BandTable.GREEN)
        band_4 = self.get_band_pixel_values(BandTable.YELLOW)

        self.processed_raster = np.divide(np.subtract(band_4, band_3), np.add(band_4, band_3))

        return self.processed_raster, self.perform_stretch()


class BuiltUp(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'built_up_index', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.COASTAL, BandTable.RED_EDGE]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_1 = self.get_band_pixel_values(BandTable.COASTAL)
        band_6 = self.get_band_pixel_values(BandTable.RED_EDGE)

        ds_temp_1 = np.subtract(band_1, band_6)
        ds_temp_2 = np.add(band_1, band_6)

        self.processed_raster = np.where(ds_temp_2 > 0, np.divide(ds_temp_1, ds_temp_2), float('nan'))

        return self.processed_raster, self.perform_stretch()


class NormalizedDifferentialVegetation(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'ndvi', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.RED, BandTable.NEAR_INFRARED_1]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_5 = self.get_band_pixel_values(BandTable.RED)
        band_7 = self.get_band_pixel_values(BandTable.NEAR_INFRARED_1)

        temp_ds_1 = np.subtract(band_7, band_5)
        temp_ds_2 = np.add(band_7, band_5)

        check = np.logical_or(temp_ds_1 > 0, temp_ds_2 > 0)
        self.processed_raster = np.where(temp_ds_2 > 0, np.divide(temp_ds_1, temp_ds_2), np.nan)

        return self.processed_raster, self.perform_stretch()


class NormalizedDifferentialVegetationRedEdge(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'ndvi_re', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.RED, BandTable.RED_EDGE]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_5 = self.get_band_pixel_values(BandTable.RED)
        band_6 = self.get_band_pixel_values(BandTable.RED_EDGE)

        temp_ds_1 = np.subtract(band_6, band_5)
        temp_ds_2 = np.add(band_6, band_5)

        check = np.logical_or(temp_ds_1 > 0, temp_ds_2 > 0)
        self.processed_raster = np.where(temp_ds_2 > 0, np.divide(temp_ds_1, temp_ds_2), np.nan)

        return self.processed_raster, self.perform_stretch()


class WorldViewWater(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'wv_water_index', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.COASTAL, BandTable.NEAR_INFRARED_2]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_1 = self.get_band_pixel_values(BandTable.COASTAL)
        band_8 = self.get_band_pixel_values(BandTable.NEAR_INFRARED_2)

        check = np.logical_or(band_1 > 0, band_8 > 0)
        self.processed_raster = np.where(check, ((band_1 - band_8) / (band_1 + band_8)), -9)

        return self.processed_raster, self.perform_stretch()


class WV3Carbonate(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'wv3_carbonate', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.SWIR_6, BandTable.SWIR_7, BandTable.SWIR_8]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_14 = self.get_band_pixel_values(BandTable.SWIR_6)
        band_15 = self.get_band_pixel_values(BandTable.SWIR_7)
        band_16 = self.get_band_pixel_values(BandTable.SWIR_8)

        self.processed_raster = np.divide(band_14, np.add(band_15, band_16))

        return self.processed_raster, self.perform_stretch()


class AluniteKaolinite(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'alunite_kaolinite', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.SWIR_3, BandTable.SWIR_5, BandTable.SWIR_6]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_11 = self.get_band_pixel_values(BandTable.SWIR_3)
        band_13 = self.get_band_pixel_values(BandTable.SWIR_5)
        band_14 = self.get_band_pixel_values(BandTable.SWIR_6)

        self.processed_raster = np.divide(np.add(band_11, band_14), band_13)

        return self.processed_raster, self.perform_stretch()


class AIOHGroupContent(Indexes):
    def __init__(self, raster_path: str, stretch_type=StretchTypes.NONE):
        Indexes.__init__(self, raster_path, 'aioh_group_content', stretch_type)

    def which_bands_required(self) -> list:
        return [BandTable.SWIR_5, BandTable.SWIR_6, BandTable.SWIR_7]

    def process_index(self) -> (np.ndarray, np.ndarray):
        band_13 = self.get_band_pixel_values(BandTable.SWIR_5)
        band_14 = self.get_band_pixel_values(BandTable.SWIR_6)
        band_15 = self.get_band_pixel_values(BandTable.SWIR_7)

        self.processed_raster = np.divide(np.add(band_13, band_15), band_14)

        return self.processed_raster, self.perform_stretch()
