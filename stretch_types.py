import logging
import math
import numpy as np


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
