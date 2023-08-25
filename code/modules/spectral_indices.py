"""
    A set of indexes used on the image matrices
"""
import numpy

from configurations import configuration
from modules import utils


def compute_normalized_difference(matrix_a, matrix_b):
    """
        Compute the normalized difference among the two matrices (matrix_a - matrix_b) / (matrix_a + matrix_b)
        :param matrix_a: The matrix representing the first matrix
        :type matrix_a: numpy.ndarray
        :param matrix_b: The matrix representing the second matrix
        :type matrix_b: numpy.ndarray
        :return: The 2D matrix representing the normalized difference index
        :rtype: numpy.ndarray
    """
    assert isinstance(matrix_a, numpy.ndarray),\
        "Wrong input parameter type for 'matrix_a', expected numpy.ndarray"

    assert isinstance(matrix_b, numpy.ndarray), \
        "Wrong input parameter type for 'matrix_b', expected numpy.ndarray"

    utils.check_matrices_same_dimensions(matrix_a, matrix_b)

    matrix_a_float = matrix_a.astype(dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    result_matrix = (matrix_a_float - matrix_b) / (matrix_a_float + matrix_b + 1e-9)
    result_matrix = matrix_a.astype(dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    return result_matrix


def compute_proportion(matrix_a, matrix_b):
    """
        Compute the proportion among the two matrices matrix_a / matrix_b
        :param matrix_a: The matrix representing the first matrix
        :type matrix_a: numpy.ndarray
        :param matrix_b: The matrix representing the second matrix
        :type matrix_b: numpy.ndarray
        :return: The 2D matrix representing the normalized difference index
        :rtype: numpy.ndarray
    """
    assert isinstance(matrix_a, numpy.ndarray),\
        "Wrong input parameter type for 'matrix_a', expected numpy.ndarray"

    assert isinstance(matrix_b, numpy.ndarray), \
        "Wrong input parameter type for 'matrix_b', expected numpy.ndarray"

    utils.check_matrices_same_dimensions(matrix_a, matrix_b)

    matrix_a = matrix_a.astype(dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    result_matrix = matrix_a / (matrix_b + 1e-9)
    result_matrix = matrix_a.astype(dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    return result_matrix


def get_ndvi_matrix(image_matrix):
    """
        Compute the NDVI (Normalized Difference Vegetation Index) given the matrix
        :param image_matrix: The matrix representing a multispectral image. The Green channel, is expected to be the
        third channel (on the matrix, the index 2). The NIR (Near InfraRed) channel, is expected to be the seventh
        channel (on the matrix, the index 6).
        :type image_matrix: numpy.ndarray
        :return: The 2D matrix representing the NDVI computed for each matrix element
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray),\
        "Wrong input parameter type for 'image_matrix', expected numpy.ndarray"

    assert image_matrix.shape[2] >= 7, "The given matrix is expected to have at least seven channels (or bands)"

    # Note: remember Python uses 0-based array
    green_matrix = image_matrix[:, :, 2]
    nir_matrix = image_matrix[:, :, 6]
    return compute_normalized_difference(nir_matrix, green_matrix)


def get_ndsi_matrix(image_matrix):
    """
        Compute the NDSI (Normalized Difference Snow Index) given the matrix
        :param image_matrix: The matrix representing a multispectral image. The Blue channel, is expected to be the
        second channel (on the matrix, the index 1). The Water Vapour channel, is expected to be the nineth
        channel (on the matrix, the index 8).
        :type image_matrix: numpy.ndarray
        :return: The 2D matrix representing the NDSI computed for each matrix element
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray),\
        "Wrong input parameter type for 'image_matrix', expected numpy.ndarray"

    assert image_matrix.shape[2] >= 9, "The given matrix is expected to have at least nine channels (or bands)"

    # Note: remember Python uses 0-based array
    green_matrix = image_matrix[:, :, 1]
    water_vapour_matrix = image_matrix[:, :, 8]
    return compute_normalized_difference(green_matrix, water_vapour_matrix)


def get_ndwi_matrix(image_matrix):
    """
        Compute the NDWI (Normalized Difference Water Index) given the matrix
        :param image_matrix: The matrix representing a multispectral image. The Aerosol channel, is expected to be the
        first channel (on the matrix, the index 0). The Water Vapour channel, is expected to be the nineth
        channel (on the matrix, the index 8).
        :type image_matrix: numpy.ndarray
        :return: The 2D matrix representing the NDWI computed for each matrix element
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray),\
        "Wrong input parameter type for 'image_matrix', expected numpy.ndarray"

    assert image_matrix.shape[2] >= 9, "The given matrix is expected to have at least nine channels (or bands)"

    # Note: remember Python uses 0-based array
    aerosol_matrix = image_matrix[:, :, 0]
    water_vapour_matrix = image_matrix[:, :, 8]
    return compute_proportion(aerosol_matrix, water_vapour_matrix)


def get_rock_sand_index_matrix(image_matrix):
    """
        Compute the Rock Sand index given the matrix
        :param image_matrix: The matrix representing a multispectral image. The NIR (Near InfraRed), is expected to be
        the seventh channel (on the matrix, the index 6). The Water Vapour channel, is expected to be the nineth
        channel (on the matrix, the index 8).
        :type image_matrix: numpy.ndarray
        :return: The 2D matrix representing the NDWI computed for each matrix element
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray),\
        "Wrong input parameter type for 'image_matrix', expected numpy.ndarray"

    assert image_matrix.shape[2] >= 9, "The given matrix is expected to have at least nine channels (or bands)"

    # Note: remember Python uses 0-based array
    nir_matrix = image_matrix[:, :, 6]
    water_vapour_matrix = image_matrix[:, :, 8]
    return compute_proportion(nir_matrix, water_vapour_matrix)


def get_evi_matrix(image_matrix):
    """
        Compute the EVI (Enhanced Vegetation Index) given the matrix
        :param image_matrix: The matrix representing a multispectral image. The NIR (Near InfraRed), is expected to be
        the seventh channel (on the matrix, the index 6). The Green channel, is expected to be the third
        channel (on the matrix, the index 2).
        :type image_matrix: numpy.ndarray
        :return: The 2D matrix representing the NDWI computed for each matrix element
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray),\
        "Wrong input parameter type for 'image_matrix', expected numpy.ndarray"

    assert image_matrix.shape[2] >= 7, "The given matrix is expected to have at least nine channels (or bands)"

    # Note: remember Python uses 0-based array
    nir_matrix = image_matrix[:, :, 6].astype(dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    red_matrix = image_matrix[:, :, 2]

    utils.check_matrices_same_dimensions(nir_matrix, red_matrix)

    result_matrix = 2.5 * (nir_matrix - red_matrix) / (nir_matrix + 2.4 * red_matrix + 10000)
    result_matrix = result_matrix.astype(dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    return result_matrix
