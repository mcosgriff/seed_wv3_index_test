import os.path
from enum import Enum

import numpy as np
import rasterio
from rasterio.enums import Compression

from stretch_types import linear_percent_stretch


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


class Indexes:
    def __init__(self, raster_path: str, index_name: str):
        self.index_name = index_name
        self.raster_path = raster_path
        self.processed_raster = None

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

    def process_index(self) -> np.ndarray:
        pass

    def build_output_profile(self) -> property:
        profile = self.raster.profile
        profile.update(count=1, dtype=rasterio.float32, compress=Compression.packbits.value)

        return profile

    def save_output(self, profile: property, output_ds: np.ndarray) -> str:
        with rasterio.open(os.path.splitext(self.raster_path)[0] + '_{}.tif'.format(self.index_name), mode='w',
                           **profile) as output:
            output.write(output_ds, 1)

            return output.name

    def index_name(self):
        return self.index_name


class Polymer1(Indexes):
    def __init__(self, raster_path: str):
        Indexes.__init__(self, raster_path, 'polymer_1_index')

    def which_bands_required(self) -> list:
        return [BandTable.SWIR_2, BandTable.SWIR_4, BandTable.SWIR_5, BandTable.SWIR_8]

    def process_index(self) -> str:
        band_10 = self.get_band_pixel_values(BandTable.SWIR_2)
        band_12 = self.get_band_pixel_values(BandTable.SWIR_4)
        band_13 = self.get_band_pixel_values(BandTable.SWIR_5)
        band_16 = self.get_band_pixel_values(BandTable.SWIR_8)

        np.seterr(divide='ignore', invalid='ignore')

        temp_ds_1 = np.where(band_12 > 0, np.divide(band_10, band_12), np.nan)
        temp_ds_2 = np.where(band_16 > 0, np.divide(band_13, band_16), np.nan)

        check = np.logical_or(temp_ds_1 != np.nan, temp_ds_2 != np.nan)

        self.processed_raster = np.where(check, temp_ds_1 + temp_ds_2, np.nan)

        return self.save_output(self.build_output_profile(), self.processed_raster)


class Polymer2(Indexes):
    def __init__(self, raster_path: str):
        Indexes.__init__(self, raster_path, 'polymer_2_index')

    def which_bands_required(self) -> list:
        return [BandTable.SWIR_2, BandTable.SWIR_3, BandTable.SWIR_6, BandTable.SWIR_8]

    def process_index(self) -> str:
        band_10 = self.get_band_pixel_values(BandTable.SWIR_2)
        band_11 = self.get_band_pixel_values(BandTable.SWIR_3)
        band_14 = self.get_band_pixel_values(BandTable.SWIR_6)
        band_16 = self.get_band_pixel_values(BandTable.SWIR_8)

        self.processed_raster = np.add(np.divide(band_10, band_11), np.divide(band_14, band_16))

        return self.save_output(self.build_output_profile(), linear_percent_stretch(self.processed_raster))


class Soil(Indexes):
    def __init__(self, raster_path: str):
        Indexes.__init__(self, raster_path, 'soil_index')

    def which_bands_required(self) -> list:
        return [BandTable.GREEN, BandTable.YELLOW]

    def process_index(self) -> str:
        band_3 = self.get_band_pixel_values(BandTable.GREEN)
        band_4 = self.get_band_pixel_values(BandTable.YELLOW)

        self.processed_raster = np.divide(np.subtract(band_4, band_3), np.add(band_4, band_3))

        return self.save_output(self.build_output_profile(), linear_percent_stretch(self.processed_raster))


class BuiltUp(Indexes):
    def __init__(self, raster_path: str):
        Indexes.__init__(self, raster_path, 'built_up_index')

    def which_bands_required(self) -> list:
        return [BandTable.COASTAL, BandTable.RED_EDGE]

    def process_index(self) -> str:
        band_1 = self.get_band_pixel_values(BandTable.COASTAL)
        band_6 = self.get_band_pixel_values(BandTable.RED_EDGE)

        ds_temp_1 = np.subtract(band_1, band_6)
        ds_temp_2 = np.add(band_1, band_6)

        self.processed_raster = np.where(ds_temp_2 > 0, np.divide(ds_temp_1, ds_temp_2), float('nan'))

        return self.save_output(self.build_output_profile(), self.processed_raster)


class NormalizedDifferentialVegetation(Indexes):
    def __init__(self, raster_path: str):
        return Indexes.__init__(self, raster_path, 'ndvi')

    def which_bands_required(self) -> list:
        return [BandTable.RED, BandTable.NEAR_INFRARED_1]

    def process_index(self) -> str:
        band_5 = self.get_band_pixel_values(BandTable.RED)
        band_7 = self.get_band_pixel_values(BandTable.NEAR_INFRARED_1)

        temp_ds_1 = np.subtract(band_7, band_5)
        temp_ds_2 = np.add(band_7, band_5)

        check = np.logical_or(temp_ds_1 > 0, temp_ds_2 > 0)
        self.processed_raster = np.where(temp_ds_2 > 0, np.divide(temp_ds_1, temp_ds_2), np.nan)

        return self.save_output(self.build_output_profile(), self.processed_raster)


class NormalizedDifferentialVegetationRedEdge(Indexes):
    def __init__(self, raster_path: str):
        Indexes.__init__(self, raster_path, 'ndvi_re')

    def which_bands_required(self) -> list:
        return [BandTable.RED, BandTable.RED_EDGE]

    def process_index(self) -> str:
        band_5 = self.get_band_pixel_values(BandTable.RED)
        band_6 = self.get_band_pixel_values(BandTable.RED_EDGE)

        temp_ds_1 = np.subtract(band_6, band_5)
        temp_ds_2 = np.add(band_6, band_5)

        check = np.logical_or(temp_ds_1 > 0, temp_ds_2 > 0)
        output_ds = np.where(temp_ds_2 > 0, np.divide(temp_ds_1, temp_ds_2), np.nan)

        return self.save_output(self.build_output_profile(), output_ds)


class WorldViewWater(Indexes):
    def __init__(self, raster_path: str):
        Indexes.__init__(self, raster_path, 'wv_water_index')

    def which_bands_required(self) -> list:
        return []

    def process_index(self) -> np.ndarray:
        band_1 = self.get_band_pixel_values(BandTable.COASTAL)
        band_8 = self.get_band_pixel_values(BandTable.NEAR_INFRARED_2)

        check = np.logical_or(band_1 > 0, band_8 > 0)
        self.processed_raster = np.where(check, ((band_1 - band_8) / (band_1 + band_8)), -9)

        return self.save_output(self.build_output_profile(), self.processed_raste)
