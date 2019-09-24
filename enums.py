from enum import Enum, auto


class Index(Enum):
    NDVI = auto()
    WORLD_VIEW_WATER = auto()
    POLYMER_1 = auto()
    POLYMER_2 = auto()
    SOIL = auto()
    BUILT_UP = auto()
    NDVI_RE = auto()
    WV3_CARBONATE = auto()
    ALUNITE_KAOLINITE = auto()
    AIOH_GROUP_CONTENT = auto()


class StretchTypes(Enum):
    LINEAR_PERCENT_STRETCH = auto()
    IMAGE_HISTOGRAM_EQUALIZATION = auto()
    NONE = auto()


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
