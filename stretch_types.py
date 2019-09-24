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


def image_histogram_equalization(image: np.ndarray, number_bins=256):
    """

    :param image:
    :param number_bins:
    :return:
    """
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
