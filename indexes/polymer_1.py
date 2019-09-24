import rasterio


class Polymer1:
    def __init__(self, raster: rasterio.DatasetReader):
        self.raster = raster
