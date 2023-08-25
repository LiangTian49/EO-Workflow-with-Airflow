import logging
import os
import multiprocessing
import re
import time
import traceback
from itertools import chain, combinations
from prettytable import PrettyTable 
import numpy
try:
    import gdal
except ImportError:
    #in Stage2022
    from osgeo import gdal
import paramiko
import skimage
from scipy import ndimage, stats
from skimage.morphology import disk, square

from configurations import configuration
from modules.exceptions.SeomException import SeomException
from modules.models.application_context import ApplicationContext
from modules.models.input_parameters import InputParameters
from modules.models.sentinel2_images import Sentinel2Images
from modules.models.seom_clc_map_legend import SeomClcMapLegend

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
def check_matrices_same_dimensions(matrix_a, matrix_b, logger=None):
    """
        Check the two matrices to have the same dimensions
        :param matrix_a: The first matrix
        :type matrix_a: numpy.ndarray
        :param matrix_b: The second matrix
        :type matrix_b: numpy.ndarray
        :return: True in case all the dimensions are equal
        :rtype: bool
    """
    assert isinstance(matrix_a, numpy.ndarray),\
        "Wrong input parameter type for 'matrix_a', expected numpy.ndarray"

    assert isinstance(matrix_b, numpy.ndarray), \
        "Wrong input parameter type for 'matrix_b', expected numpy.ndarray"

    if matrix_a.size != matrix_b.size:
        raise SeomException("The two matrices have different size")

    if len(matrix_a.shape) != len(matrix_b.shape):
        raise SeomException("The two matrices have different dimensions")

    for dimension in range(0, len(matrix_a.shape)):
        if matrix_a.shape[dimension] != matrix_b.shape[dimension]:
            raise SeomException("The two matrices have different size in dimension " + str(dimension) + " (zero based)")

    return True


def __compute_majority_rule_process(matrix, identifier, result_queue):
    """
        The computation for the majority rule in the matrix for a single process in a multi-process execution
        :param matrix: The matrix to compute the majority rule on
        :type matrix: numpy.ndarray
        :param identifier: The current process identifier
        :type identifier: int
        :param result_queue: The queue to store the result
        :type result_queue: multiprocessing.Queue
    """
    assert isinstance(matrix, numpy.ndarray),\
        "Wrong input parameter type for 'matrix', expected numpy.ndarray"

    # Computing the mode (returns a tuple of modes and counts)
    result = stats.mode(matrix, axis=1)

    # Returning the modes
    result_queue.put((identifier, result[0][:, 0]))


def compute_majority_rule(matrix, parallelize=False, processors_count=None, logger=None):
    """
        Perform the majority rule (mode) on the matrix along the columns.
        :param matrix: The matrix to compute the majority rule on
        :type matrix: numpy.ndarray
        :param parallelize: Flag for activating the parallel computation
        :type parallelize: bool
        :param processors_count: The number of processors required (up to the available). In case not provided, all the
        available processors will be used.
        :type processors_count: int
        :param logger: The logger
        :type logger: logging.Logger
        :return: The moded matrix having just one column
        :rtype: numpy.ndarray
    """
    assert isinstance(matrix, numpy.ndarray),\
        "Wrong input parameter type for 'matrix', expected numpy.ndarray"
    assert matrix.ndim == 2, "The majority rule operation works only on 2D matrixes"

    # The available processors
    available_processors = get_processors_count()
    if processors_count is not None and processors_count < available_processors:
        available_processors = processors_count

    if not parallelize or available_processors == 1:
        # Computing the mode (returns a tuple of modes and counts)
        result = stats.mode(matrix, axis=1)

        # Returning the modes
        return result[0][:, 0]

    # Separate computation accordingly with processors available
    process_slot_size = int(matrix.shape[0] / available_processors)
    active_processes = []
    result_queue = multiprocessing.Queue()
    start_index = 0
    for p in range(0, available_processors):
        end_index = start_index + process_slot_size
        sub_matrix = None

        # In case the end is outbound or it's the last processor
        if end_index > matrix.shape[0] or p == (available_processors - 1):
            sub_matrix = matrix[start_index:, :]
        else:
            sub_matrix = matrix[start_index:end_index, :]

        process = multiprocessing.Process(
            target=__compute_majority_rule_process,
            args=(sub_matrix, p, result_queue))
        active_processes.append(process)
        process.start()
        start_index = end_index

    # Acquiring results
    results = dict()
    while len(active_processes) > 0:
        # Just waiting
        time.sleep(10)

        # Extract the results from queue before join for avoiding deadlock
        while result_queue.qsize() > 0:
            single_result = result_queue.get()
            results[single_result[0]] = single_result[1]

        # Removing processes
        for process in active_processes:
            # Skip the process if it is still working
            if process.is_alive():
                continue

            process.join()
            process.close()
            active_processes.remove(process)

    # Aggregating results
    final_result = None
    for p in range(0, available_processors):
        if final_result is None:
            final_result = results[p]
            continue

        final_result = numpy.concatenate((final_result, results[p]), axis=0)

    return final_result


def compute_powerset(iterable):
    """
        Compute the powerset of the given iterable.
        e.g. powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        From https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def dilate_matrix_binary(original_matrix, radius=2, type='disk', logger=None):
    """
        Perform the dilate on the given matrix
        :param original_matrix: The matrix to dilate
        :type original_matrix: numpy.ndarray
        :param radius: The dimension for the disk structure to apply
        :type radius: int
        :return: The dilated matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(original_matrix, numpy.ndarray) \
        , "Wrong input parameter type for 'original_matrix', expected numpy.ndarray"

    # Definition of the eroding structure
    dilate_mask = None
    if type.lower() == 'disk':
        dilate_mask = disk(radius=radius, dtype=numpy.uint8)
    elif type.lower() == 'square':
        dilate_mask = square(width=radius, dtype=numpy.uint8)
    else:
        raise SeomException("The given parameter 'type' is {} which is not yet implemented".format(type))

    # Dilate
    result_matrix = ndimage.binary_dilation(original_matrix, structure=dilate_mask).astype(original_matrix.dtype)

    return result_matrix


def erode_matrix_binary_same_values(original_matrix, disk_radius=2, logger=None):
    """
        Perform the erosion on the given matrix, considering each group of values together (e.g. in the CLC a group
        is a class ~ "Urban"...)
        :param original_matrix: The matrix to erode
        :type original_matrix: numpy.ndarray
        :param disk_radius: The dimension for the disk structure to apply
        :type disk_radius: int
        :return: The eroded matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(original_matrix, numpy.ndarray)\
        , "Wrong input parameter type for 'original_matrix', expected numpy.ndarray"

    # Definition of the eroding structure
    erosion_mask = disk(radius=disk_radius, dtype=numpy.uint8)
    # Redefinition as matrix of booleans
    erosion_mask = erosion_mask == 1

    # The resulting matrix
    result_matrix = numpy.zeros((original_matrix.shape[0], original_matrix.shape[1]),
                                dtype=original_matrix.dtype)

    reshaped_original_matrix = original_matrix.reshape(original_matrix.shape[0], original_matrix.shape[1])

    # Retrieving the unique class code values from the matrix (expected to be the CLC)
    unique_class_values = numpy.unique(original_matrix)

    # Iterate over all the converted classes
    for class_value in unique_class_values:
        # Defining the masking as a matrix of booleans
        class_mask = reshaped_original_matrix == class_value

        # Eroding the mask
        eroded_class_mask = ndimage.binary_erosion(class_mask, structure=erosion_mask)

        # Setting the class code using the eroded mask
        result_matrix[eroded_class_mask] = class_value

    # To be consistent in the future steps, explicitly define a band
    result_matrix = result_matrix.reshape(result_matrix.shape[0], result_matrix.shape[1], 1)
    return result_matrix


def erode_matrix_small_objects_same_values(original_matrix, minimum_size=2, logger=None):
    """
        Perform the erosion on the given matrix, removing all connected objects smaller than the given size,
        considering each group of values together (e.g. in the CLC a group is a class ~ "Urban"...)
        :param original_matrix: The matrix to erode
        :type original_matrix: numpy.ndarray
        :param minimum_size: The minimum size of the objects to keep
        :type minimum_size: int
        :return: The eroded matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(original_matrix, numpy.ndarray) \
        , "Wrong input parameter type for 'original_matrix', expected numpy.ndarray"

    # The resulting matrix
    result_matrix = numpy.zeros((original_matrix.shape[0], original_matrix.shape[1]),
                                dtype=original_matrix.dtype)

    # Retrieving the unique class code values from the matrix (expected to be the CLC)
    unique_class_values = numpy.unique(original_matrix)

    # Iterate over all the converted classes
    for class_value in unique_class_values:
        # Defining the masking as a matrix of booleans
        class_mask = original_matrix == class_value

        # Eroding the mask (use in-place because of huge memory consumption)
        #skimage.morphology.remove_small_objects(class_mask, min_size=minimum_size, connectivity=1, in_place=True)
        skimage.morphology.remove_small_objects(class_mask, min_size=minimum_size, connectivity=1)

        # Setting the class code using the eroded mask
        result_matrix[class_mask] = class_value

    # To be consistent in the future steps, explicitly define a band
    result_matrix = result_matrix.reshape(result_matrix.shape[0], result_matrix.shape[1], 1)
    return result_matrix


def get_clouds_level_matrix(application_context, filter_names=[]):
    """
        Compute the clouds level matrix, reporting in each pixel the number of clouds above in the Sentinel2 data
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param filter_names: The list of image names to consider. If None or empty all images will be considered.
        :type filter_names: list of str
        :return: The matrix reporting the clouds level over each pixels, wrt the Sentinel2 data
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"

    x_size = application_context.s2_images.x_size
    y_size = application_context.s2_images.y_size

    # The matrix represeting the clouds level
    result_matrix = numpy.zeros((x_size, y_size, 1), dtype=numpy.uint8)

    for image_name in application_context.s2_images.name_cloud_covers_matrix_map:
        if filter_names is not None and len(filter_names) > 0 and image_name not in filter_names:
            continue

        image_cmask = application_context.s2_images.name_cloud_covers_matrix_map[image_name]
        result_matrix[image_cmask > 0] += 1

    return result_matrix


def get_date_components_by_text(text):
    """
        From a given text retrieve the date components from the expected configured pattern (e.g. YYYYMMdd)
        :param text: The text contaning the date (e.g. 20180419)
        :type text: str
        :return: The tuple of year, month and day
        :rtype: int, int, int
    """
    assert isinstance(text, str), "Wrong input parameter type for 'text', expected str"

    # Composing the regex for retrieving the full date, the month and the day from the file name like to be:
    file_name_date_regex = re.compile(configuration.DATE_REGEX)

    # Using regular expression for retrieving the date components
    # The result will be: complete date (e.g. 20180419), the year (e.g. 2018), the month (e.g. 04) and the day (e.g.
    # 19).
    regex_matches = file_name_date_regex.match(text)
    if regex_matches is None or regex_matches.groups() is None or len(regex_matches.groups()) < 4:
        raise SeomException("Unable to retrieve the expected matches groups (expected 4) from the given text " + text)

    # Getting the year
    tile_year = int(regex_matches.groups()[1])
    tile_month = int(regex_matches.groups()[2])
    tile_day = int(regex_matches.groups()[3])

    return tile_year, tile_month, tile_day


def get_feature_selection_by_image_name(application_context, s2_image_name, feature_set=None, absolute=False):
    """
        Retrieve the feature selection for the given image, considering either the given one or the one in configuration
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param s2_image_name: The name of the image the feature set to compute is related to
        :type s2_image_name: str
        :param feature_set: List of bands for all the images. In case not provided, the one SVM_FEATURE_SET in
        configuration will be used
        :type feature_set: list of int
        :param absolute: If true, it refers the indexes in the feature_set. If false, it will refer the image relative
        features (e.g. 0-9)
        :type absolute: bool
        :return: The list of feature (i.e. bands) for the image required by the feature_set; it is a subset of the
        feature_set or the one SVM_FEATURE_SET in configuration
        :rtype: list of int or None if no feature selected for the image
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"

    if feature_set is None:
        feature_set = configuration.SVM_FEATURE_SET

    # Get total count for checking the feature_set values
    s2_images_count = len(application_context.s2_images.name_matrix_map)
    assert numpy.max(feature_set) < s2_images_count * application_context.s2_images.bands_size, \
        "The maximum number in the feature_set is greater than the number of features of all the images"

    # Get the theoretical 'offset' for the given image
    image_offset = list(application_context.s2_images.name_matrix_map.keys()).index(s2_image_name)

    # Using the S2 bands size for computing the theoretical bands for the images
    bands_size = application_context.s2_images.bands_size
    image_band_start = image_offset * bands_size
    image_band_end = image_band_start + bands_size - 1

    # Retrieving the feature selection for the given image
    feature_set_subset = [f for f in feature_set if image_band_start <= f <= image_band_end]

    if len(feature_set_subset) == 0:
        return [10]

    if absolute:
        return feature_set_subset

    result = list(numpy.array(feature_set_subset) - image_band_start)
    return result


def get_less_cloudy_cmasks(application_context, logger=None):
    """
        Retrieve the less cloudy cmasks
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param logger: The logger
        :type logger: logging.Logger
        :return: The cmasks names in ordered list from less cloudy to most cloudy
        :rtype: list of str
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"

    # Compute the clouds count
    cloud_level_map = dict()
    for cloud_image_name in application_context.s2_images.name_cloud_covers_matrix_map:
        cloud_image = application_context.s2_images.name_cloud_covers_matrix_map[cloud_image_name]

        # Using the > 0 for having a flexible way in case the cmasks may change reported information
        cloud_level_map[cloud_image_name] = numpy.sum(cloud_image[cloud_image > 0])

    # Sorting
    sorted_cloud_level_map = sorted(cloud_level_map.items(), key=lambda key_value: key_value[1], reverse=False)

    # Getting the names
    less_cloudy_sorted_cloud_masks = [element[0] for element in sorted_cloud_level_map]

    return less_cloudy_sorted_cloud_masks


def get_random_sample(matrix, sample_number, axis=0, indexes=True):
    """
        Retrieve a sample of the given matrix
        :param matrix: The input matrix to sample
        :type matrix: numpy.ndarray
        :param sample_number: The number of elements (or indexes) to extract
        :type sample_number: int
        :param axis: The axis dimension to sample
        :type axis: int
        :param indexes: True in case the return are the indexes for the sampling. If False, actual values are returned
        :type indexes: bool
        :return:
    """
    assert isinstance(matrix, numpy.ndarray), "Wrong input parameter type for 'matrix', expected numpy.ndarray"
    assert matrix.ndim >= axis + 1, "The requested axis is out of the matrix dimensions"
    assert matrix.shape[axis] >= sample_number, "The requested samples for the axis is greater than available values"

    # Performing the sample (i.e. are the indexes along the axis)
    sample = numpy.random.random_integers(0, matrix.shape[axis] - 1, sample_number)
    if indexes:
        return sample

    if axis == 0:
        return matrix[sample]

    if axis == 1:
        return matrix[:, sample]

    if axis == 2:
        return matrix[:, :, sample]

    if axis == 3:
        return matrix[:, :, :, sample]


def get_s2_cloud_masks_images_file_paths_map(application_context):
    """
        Retrieve the map for the Sentinel2 corresponding cloud masks.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :return: The map having as key the name of the Sentinel2 image and as value the path of the corresponding cloud
        mask
        :rtype: dict
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"

    # The cloud mask images path from which retrieve the images
    cloud_masks_folder_path = os.path.dirname(application_context.input_parameters.cloud_mask_image_path)

    # Composing the regex for retrieving the full date, the month and the day from the file name like to be:
    # MSIL2A_20180926T101021_N0208_R022_T32TPS.tif
    file_name_date_regex = re.compile(configuration.IMAGES_S2_FILE_NAME_DATE_REGEX)

    # The map having as key the Sentinel2 image name and as value the path to its corresponding cloud mask image
    s2_name_cloud_mask_path_map = dict()

    # Iterate over all the S2 images available
    for s2_image_name in application_context.s2_images.name_matrix_map:
        # Using regular expression for retrieving the date components from the Sentinel2 image name
        # The result will be: complete date (e.g. 20180419), the year (e.g. 2018), the month (e.g. 04) and the day (e.g.
        # 19).
        regex_matches = file_name_date_regex.match(s2_image_name)
        if regex_matches is None or regex_matches.groups() is None or len(regex_matches.groups()) < 4:
            raise SeomException("Unable to retrieve the expected matches groups (expected 4) from Sentinel2 file name")

        full_date = int(regex_matches.groups()[0])
        cloud_mask_file_name = \
            configuration.CLOUD_MASK_NAME_PREFIX + str(full_date) + configuration.CLOUD_CMASK_NAME_SUFFIX
        s2_name_cloud_mask_path_map[s2_image_name] = os.path.join(cloud_masks_folder_path, cloud_mask_file_name)

    return s2_name_cloud_mask_path_map


def get_images_paths_from_folder(folder_path, logger=None):
    """
        Retrieve all the images (e.g. *.tif - actual extensions set in configuration) paths from the given folder path
        :param folder_path: The folder path containing the images
        :type folder_path: str
        :return: a list of images in the path
        :rtype: list of str
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise SeomException("The given path doesn't exist or it is not a directory")

    result = []
    for element in sorted(os.listdir(folder_path)):
        if is_feasible_image_file(element):
            result.append(os.path.join(folder_path, element))

    return result


def get_patches_coordinates(x_size, y_size, patch_size):
    """
        Retrieve the patch coordinates considering the x_size, y_size (width and height of a target image)
        and the configuration
        :param x_size: The image width (pixels)
        :type x_size: int
        :param y_size: The image height (pixels)
        :type y_size: int
        :param patch_size: The size of each patch in pixels (expected to be a divider of both x_size and y_size)
        :type patch_size: int
        :return: Array of tuple, each one having x_start, x_end, y_start, y_end
        :rtype: (int, int, int, int)
    """
    assert isinstance(x_size, int), "Wrong type for 'x_size', int"
    assert isinstance(y_size, int), "Wrong type for 'y_size', int"
    assert isinstance(patch_size, int), "Wrong type for 'patch_size', int"
    assert patch_size <= x_size, "The patch size is greater than the x_size"
    assert patch_size <= y_size, "The patch size is greater than the y_size"
    assert x_size % patch_size == 0, "The x_size is not divisible by the patch_size"
    assert y_size % patch_size == 0, "The y_size is not divisible by the patch_size"

    patches = []

    # Computing the patches to process (as array of tuples having the invocation parameters)
    x_start = 0
    # Iterating over all columns
    while x_start < x_size:
        # The patch end 'x' coordinate
        x_end = x_start + patch_size
        # Starting from the first column
        y_start = 0

        # Iterating over all rows
        while y_start < y_size:
            y_end = y_start + patch_size

            # Adding the patch
            patches.append((x_start, x_end, y_start, y_end))

            # Next 'row'
            y_start = y_start + patch_size

        # Next 'column'
        x_start = x_end

    return patches



def get_epsg_code_from_gdal_projection(projection_string):
    prj = gdal.osr.SpatialReference(wkt=projection_string)
    code = prj.GetAttrValue('AUTHORITY', 1)
    return int(code)


# TODO Move to spatial_utils
def get_map_coordinates_from_indexes(geo_transform, matrix_indexes):
    """
        Decode the matrix_indexes into map coordinates
        :param geo_transform: The geo transformation associated to the source matrix
        (e.g. (x_min, xres, 0, y_max, 0, -yres) meaning that xres is the pixel size in the x-axis and yres is the pixel
        size in the y-axis)
        :type geo_transform: tuple
        :param matrix_indexes: The tuple of indexes to decode in the form of tuple
        (row-indexes-array, column-indexes-array,..)
        :type matrix_indexes: tuple
        :return: coordinates, x_min, y_min, x_max, y_max
        :rtype:
    """
    assert isinstance(matrix_indexes, tuple) or isinstance(matrix_indexes, list), \
        "Wrong input type for parameter 'matrix_indexes', expected tuple or list"
    assert len(matrix_indexes) > 1, "Expected the 'matrix_indexes' to have at least 2 elements"
    assert isinstance(geo_transform, tuple), "Wrong input type for parameter 'geo_transform', expected tuple"
    assert len(geo_transform) == 6, "Expected the 'matrix_indexes' to have exactly 6 elements"

    # The pixel size for the matrix, which may differ slightly from configuration.IMAGES_PIXEL_RESOLUTION due
    # some reprojection involved
    pixel_size = geo_transform[1]

    # Retrieving the data from the geo transformation to be used for computation
    matrix_x_min = geo_transform[0]
    matrix_y_max = geo_transform[3]

    # The init values
    x_min = 6000000000
    y_min = 6000000000
    x_max = 0
    y_max = 0
    coordinates = []

    # Process all the indexes
    length = len(matrix_indexes[0])
    for i in range(0, length):
        row = matrix_indexes[0][i]
        column = matrix_indexes[1][i]

        # Computing the map coordinates from indexes
        x_map = matrix_x_min + column * pixel_size
        y_map = matrix_y_max - row * pixel_size

        # Adding to the list
        coordinates.append([x_map, y_map])

        # Compute the coordinates for an extent containing all the indexes
        if x_map < x_min:
            x_min = x_map

        if x_map > x_max:
            x_max = x_map

        if y_map < y_min:
            y_min = y_map

        if y_map > y_max:
            y_max = y_map

    return coordinates, x_min, y_min, x_max, y_max


def get_normalized_image_matrix(image_matrix, min_values, max_values):
    """
        Normalize the matrix considering the arrays of min_values and the arrays of max_values
        :param image_matrix: The image matrix (3D)
        :type image_matrix: numpy.ndarray
        :param min_values: The array of minimum values (column wise)
        :type min_values: numpy.ndarray
        :param max_values: The array of maximum values (column wise)
        :type max_values: numpy.ndarrayIMAGES_NORMALIZED_NUMPY_TYPE
        :return: The normalized matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray), "Wrong type for 'image_matrix', expected numpy.ndarray"
    assert isinstance(min_values, numpy.ndarray), "Wrong type for 'min_values', expected numpy.ndarray"
    assert isinstance(max_values, numpy.ndarray), "Wrong type for 'max_values', expected numpy.ndarray"

    # Normalization through min-max scaling
    result_matrix = (image_matrix - min_values) / (max_values - min_values)
    # Casting to another data type for lowering the memory use
    result_matrix = result_matrix.astype(configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    return result_matrix

def get_percentiles(image_matrix):
    print("image_matrix shape: ", image_matrix.shape)
    p1 = numpy.percentile(image_matrix, 1, axis = 0)
    p99 = numpy.percentile(image_matrix, 99, axis = 0)

    print("p1, p99", p1, p99)
    return p1, p99

def get_percentiles2d(image_matrix):
    print("image_matrix shape: ", image_matrix.shape)
    p1 = numpy.percentile(image_matrix, 1, axis = (0,1))
    p99 = numpy.percentile(image_matrix, 99, axis = (0,1))

    print("p1, p99", p1, p99)
    return p1, p99
            
def get_normalized_image_matrix_percentile(image_matrix):
    """
        Normalize the matrix considering the arrays of min_values and the arrays of max_values
        :param image_matrix: The image matrix (3D)
        :type image_matrix: numpy.ndarray
        :param min_values: The array of minimum values (column wise)
        :type min_values: numpy.ndarray
        :param max_values: The array of maximum values (column wise)
        :type max_values: numpy.ndarrayIMAGES_NORMALIZED_NUMPY_TYPE
        :return: The normalized matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray), "Wrong type for 'image_matrix', expected numpy.ndarray"
    #clip matrix between 1 and 99 percentiles
    p1, p99 = get_percentiles(image_matrix)
    image_matrix = numpy.clip(image_matrix, a_min = p1, a_max = p99)
    std = image_matrix.std(axis = 0)
    mean = image_matrix.mean(axis = 0)
    # Normalization through zero mean 1 std
    result_matrix = (image_matrix - mean) / std
    # Casting to another data type for lowering the memory use
    result_matrix = result_matrix.astype(configuration.IMAGES_NORMALIZED_NUMPY_TYPE)

    return result_matrix

def get_normalized_image_matrix2d_percentile(image_matrix):
    """
        Normalize the matrix considering the arrays of min_values and the arrays of max_values
        :param image_matrix: The image matrix (3D)
        :type image_matrix: numpy.ndarray
        :param min_values: The array of minimum values (column wise)
        :type min_values: numpy.ndarray
        :param max_values: The array of maximum values (column wise)
        :type max_values: numpy.ndarrayIMAGES_NORMALIZED_NUMPY_TYPE
        :return: The normalized matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(image_matrix, numpy.ndarray), "Wrong type for 'image_matrix', expected numpy.ndarray"
    #clip matrix between 1 and 99 percentiles
    p1, p99 = get_percentiles2d(image_matrix)
    print("p1, p99", p1, p99)
    image_matrix = numpy.clip(image_matrix, a_min = p1, a_max = p99)
    std = image_matrix.std(axis = (0,1))
    mean = image_matrix.mean(axis = (0,1))
    # Normalization through zero mean 1 std
    result_matrix = (image_matrix - mean) / std
    # Casting to another data type for lowering the memory use
    result_matrix = result_matrix.astype(configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    print("result_matrix min:", result_matrix.min(axis=(0,1)))
    print("result_matrix max:", result_matrix.max(axis=(0,1)))
    print("result_matrix mean:", result_matrix.mean(axis=(0,1)))
    print("result_matrix std:", result_matrix.std(axis=(0,1)))

    return result_matrix

def clip_matrix(image_matrix):
    assert isinstance(image_matrix, numpy.ndarray), "Wrong type for 'image_matrix', expected numpy.ndarray"
    #clip matrix between 1 and 99 percentiles

    p1 = numpy.percentile(image_matrix, 1)
    p99 = numpy.percentile(image_matrix, 99)
    print("p1, p99", p1, p99)
    image_matrix = numpy.clip(image_matrix, a_min = p1, a_max = p99)

    # Casting to another data type for lowering the memory use
    image_matrix = image_matrix.astype(configuration.IMAGES_NORMALIZED_NUMPY_TYPE)


    return image_matrix


def normalization(img_vector,a,b):
    minBands = numpy.zeros((10)) + 1
    maxBands = numpy.zeros((10)) + 12000
    idx = numpy.where(img_vector==0)
    img_vector = img_vector.astype(configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    for i in range(img_vector.shape[1]):
        img_vector[:,i] = ((b-a)*(img_vector[:,i] - minBands[i]) / (maxBands[i] - minBands[i]) )+ a
    img_vector[idx] = 0
    return img_vector

def normalization2d(img_vector,a,b):
    minBands = numpy.zeros((10)) + 1
    maxBands = numpy.zeros((10)) + 12000
    idx = numpy.where(img_vector==0)
    img_vector = img_vector.astype(configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
    for i in range(img_vector.shape[2]):
        img_vector[:,:,i] = ((b-a)*(img_vector[:,:,i] - minBands[i]) / (maxBands[i] - minBands[i]) )+ a
    img_vector[idx] = 0
    return img_vector

def stack_bands(data_matrix):
    n_bands = 10
    num_obs = data_matrix.shape[1] // n_bands
    arr_stacked = numpy.zeros([data_matrix.shape[0] * num_obs, n_bands])
    for i in range(0, num_obs):
        arr_stacked[i * data_matrix.shape[0] : (i + 1) * data_matrix.shape[0], :] = data_matrix[:, i * n_bands : (i + 1) * n_bands]
    return arr_stacked, num_obs, n_bands

def invert_stack_bands(data_matrix, num_obs, n_bands):
    arr_inv_stacked = numpy.zeros([data_matrix.shape[0] // num_obs, num_obs * n_bands])
    for i in range(0, num_obs):
        arr_inv_stacked[:, i * n_bands : (i + 1) * n_bands] = data_matrix[i * (data_matrix.shape[0] // num_obs) : (i + 1) * (data_matrix.shape[0] // num_obs), :]
    return arr_inv_stacked

def get_processors_count():
    """
        Retrieve the number of processors of current machine.
        :return: The number of available processor for computing
        :rtype: int
    """
    return multiprocessing.cpu_count()


def get_processors_count_classification():
    """
        Retrieve the number of processors to use for classification, considering the configuration
        and the actual machine resources. It will return the lower among the two.
        :return: The number of available processor for computing
        :rtype: int
    """
    processors_count = configuration.SVM_PARALLEL_THREADS_REQUEST
    if multiprocessing.cpu_count() < processors_count:
        processors_count = multiprocessing.cpu_count()

    return processors_count


def get_processors_count_clustering():
    """
        Retrieve the number of processors to use for clustering, considering the configuration
        and the actual machine resources. It will return the lower among the two.
        :return: The number of available processor for computing
        :rtype: int
    """
    processors_count = configuration.SAMPLES_KMEANS_PARALLEL_THREADS_REQUEST
    if multiprocessing.cpu_count() < processors_count:
        processors_count = multiprocessing.cpu_count()

    return processors_count



def get_processors_count_post_classification():
    """
        Retrieve the number of processors to use for post-classification, considering the configuration
        and the actual machine resources. It will return the lower among the two.
        :return: The number of available processor for computing
        :rtype: int
    """
    processors_count = configuration.POST_CLASSIFICATION_PARALLEL_THREADS_REQUEST
    if multiprocessing.cpu_count() < processors_count:
        processors_count = multiprocessing.cpu_count()

    return processors_count


def get_remote_output_path(application_context=None, tile_year=None, tile_name=None):
    """
        Retrieve the remote output path using the data in the given application_context
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param tile_year: The year of the tile
        :type tile_year: int
        :param tile_name: The name of the tile
        :type tile_name: str
        :return: The remote path to store the results
        :rtype: str
    """
    if application_context is not None:
        assert isinstance(application_context, ApplicationContext), \
            "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"
        year = application_context.tile_year
        name = application_context.tile_name
    else:
        assert isinstance(tile_year, int), "Wrong input parameter type for 'tile_year', expected int"
        assert isinstance(tile_name, str), "Wrong input parameter type for 'tile_name', expected str"
        year = tile_year
        name = tile_name

    remote_path = os.path.join(configuration.CLASSIFICATION_REMOTE_DATA_OUTPUT_PATH,
                               str(year) + "_" + name,
                               configuration.DATA_OUTPUT_FOLDER_NAME)

    return remote_path


def get_s2_image_name_by_date(files_list, date):
    """
        Retrieve the element from the list which represent the given date
        :param files_list: The list of Sentinel2 files names (feasible even with path)
        :type files_list: list of str
        :param date: The string representing the date (e.g. '20181026')
        :type date: str
        :return: The element from the list corresponding to the date
        :rtype: str
    """
    assert isinstance(files_list, list), "Wrong input parameter for 'files_list', expected list"
    assert all(isinstance(element, str) for element in files_list), \
        "Wrong input parameter for 'files_list', expected list of str"
    assert isinstance(date, str), "Wrong input parameter for 'date', expected str"

    candidates = [element for element in files_list if date in element]
    if len(candidates) == 0:
        raise SeomException("In the given list, there isn't any file matching the given date " + str(date))

    if len(candidates) == 1:
        return candidates[0]

    # In case there are multiple candidates, check on the proper name pattern
    candidate = None
    for s2_image_name in candidates:
        # In case it isn't a match, skip the file name
        if not is_valid_sentinel2_image_name(s2_image_name):
            continue

        # In case there is already a candidate for the pattern, cannot decide which one to use (either the pattern is
        # not exaustive or there is some other issue)
        if candidate is not None:
            raise SeomException("In the given list, there are multiple files matching the given date " + str(date))

        # Saving the candidate
        candidate = s2_image_name

    return candidate


def get_s2_images_names_top_k(files_list, k):
    """
        Retrieve the top k valid Sentinel 2 images names from the list
        :param files_list: The list of Sentinel2 files names (feasible even with path)
        :type files_list: list of str
        :param k: The number of images to retrieve (ascending order from the first)
        :type k: int
        :return: The top k valid elements from the list
        :rtype: list
    """
    assert isinstance(files_list, list), "Wrong input parameter for 'files_list', expected list"
    assert all(isinstance(element, str) for element in files_list), \
        "Wrong input parameter for 'files_list', expected list of str"
    assert isinstance(k, int), "Wrong input parameter for 'k', expected str"

    # Filtering out the wrong names
    candidates = [element for element in files_list if is_valid_sentinel2_image_name(element)]
    if len(candidates) == 0:
        raise SeomException("In the given list, there isn't any valid file name")

    if len(candidates) < k:
        raise SeomException("There aren't enough elements for the request (k = {})".format(k))

    return candidates[:k]


def get_shapefile_files_path(shapefile_path):
    """
        The shapefile format consists of at least the files: *.shp, *.shx and *.dbf.
        Moreover there are also other files associated, such as *.prj.
        Given the file path for the *.shp, it will return the file paths of all the relative associated files.
        :param shapefile_path: The file path for the *.shp file
        :type shapefile_path: str
        :return: The list of file paths associated to the given one (included in the return list)
        :rtype: list of str
    """
    assert isinstance(shapefile_path, str), "Wrong input parameter type for 'shapefile_path', expected str"
    assert os.path.isfile(shapefile_path), "The 'shapefile_path' is expected to be a valid file path"
    assert len(os.path.splitext(shapefile_path)) == 2, "The 'shapefile_path' doesn't have any extension"
    assert os.path.splitext(shapefile_path)[1].lower() == ".shp", "The 'shapefile_path' doesn't refer a file *.shp"

    # Feasible extensions
    feasible_extensions = ['.shp', '.shx', '.dbf', '.prj']

    # The results
    results = []

    # Scanning the parent folder
    parent_directory = os.path.dirname(shapefile_path)
    for file_name in sorted(os.listdir(parent_directory)):
        splitted = os.path.splitext(file_name)

        # Skip those not having extension
        if len(splitted) == 1:
            continue

        if splitted[1].lower() in feasible_extensions:
            file_path = os.path.join(parent_directory, file_name)
            results.append(file_path)

    return results


def get_temporary_folder_path(application_context, temporary_path):
    """
        Retrieve the path to the locally temporary folder (e.g. for storing the dowloaded data).
        In case it doesn't exist, it will create it.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param temporary_path: The temporary path for downloading the resource
        :type temporary_path: str
        :return: The local temporary folder
        :rtype: str
    """
    # TODO Review the function considering the use of temporary folder from ApplicationContext (added later)
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"
    assert isinstance(temporary_path, str), \
        "Wrong input parameter type for 'temporary_path', expected str"
    assert os.path.exists(temporary_path), "The path {} doesn't exist".format(temporary_path)

    # Getting the name of the Sentinel2 images folder
    tile_folder_name = os.path.basename(os.path.dirname(application_context.input_parameters.s2_data_path))

    # Generating the Sentinel2 folder in the temporary folder
    temporary_folder_path = os.path.join(temporary_path, tile_folder_name)
    if not os.path.exists(temporary_folder_path):
        os.makedirs(temporary_folder_path)

    return temporary_folder_path


def get_tile_date_by_s2_file_name(file_name):
    """
        Retrieve the tiles name from the file name (e.g. MSIL2A_20180827T101021_N0208_R022_T32TPS.tif)
        :param file_name: The file name (or even the path)
        :type file_name: str
        :return: The tile name, the year
        :rtype: str, int
    """
    assert isinstance(file_name, str), "Wrong input parameter type for 'file_name', expected str"

    actual_file_name = os.path.basename(file_name)

    # Getting the tile name
    tile_name = os.path.splitext(actual_file_name)[0].split("_")[-1]

    # Composing the regex for retrieving the full date, the month and the day from the file name like to be:
    # MSIL2A_20180926T101021_N0208_R022_T32TPS.tif
    file_name_date_regex = re.compile(configuration.IMAGES_S2_FILE_NAME_DATE_REGEX)

    # Using regular expression for retrieving the date components from the Sentinel2 image name
    # The result will be: complete date (e.g. 20180419), the year (e.g. 2018), the month (e.g. 04) and the day (e.g.
    # 19).
    regex_matches = file_name_date_regex.match(actual_file_name)
    if regex_matches is None or regex_matches.groups() is None or len(regex_matches.groups()) < 4:
        raise SeomException("Unable to retrieve the expected matches groups (expected 4) from Sentinel2 file name")

    # Getting the year
    tile_year = int(regex_matches.groups()[1])
    tile_month = int(regex_matches.groups()[2])
    tile_day = int(regex_matches.groups()[3])

    return tile_name, tile_year, tile_month, tile_day


def get_unique_land_cover_class_ids(clc_map, logger=None):
    """
        Retrieve the unique values of the given Corine Land Cover map classes identifiers (e.g. [1, 2, 3])
        :param clc_map: The Corine Land Cover map
        :type clc_map: numpy.ndarray
        :return: The list of unique Corine Land Cover map classes indentifiers
        :rtype: list of int
    """
    assert isinstance(clc_map, numpy.ndarray)\
        , "Wrong input parameter type for 'clc_map', expected to be numpy.ndarray"

    result = numpy.unique(clc_map)
    return result


def initialize_legend_colors(seom_legend, color_map_path):
    """
        Initialize the given seom_legend with the colors in the color_map_path file.
        :param seom_legend: The legend to initialize
        :type seom_legend: SeomClcMapLegend
        :param color_map_path: The path for the color map file
        :type color_map_path: str
    """
    assert isinstance(seom_legend, SeomClcMapLegend), \
        "Wrong input parameter type for 'seom_legend', expected SeomClcMapLegend"

    # Aquiring the legend colors from external file (e.g. QGis file)
    legend_colors = retrieve_legend_colors(color_map_path)

    # Setting the colors on the generated legend, if not already set
    for element in seom_legend.seom_classes:
        if element.has_colors_set():
            continue

        element_colors = legend_colors[element.class_value]
        element.set_colors(element_colors[0], element_colors[1], element_colors[2], element_colors[3])


def is_feasible_image_file(file_path, logger=None):
    """
        Check if the file is feasible as image
        :param file_path: The file path
        :return: True in case the file path belongs to the feasible types (configuration)
        :rtype: bool
    """
    if file_path is None or len(file_path) == 0:
        return False

    feasible_extensions = configuration.IMAGES_FEASIBLE_EXTENSIONS

    element_parts = os.path.splitext(file_path)
    if element_parts is None or len(element_parts) < 2:
        return False

    return element_parts[1] in feasible_extensions


def is_valid_classified_image_name(name, classification_type):
    """
        Check against a configured regular expression if the given name is a valid classified image.
        That is the result for the entire pipeline.
        :param name: The candidate classified image name
        :type name: str
        :param classification_type: The kind of classified like 'trial', 'final-classified', 'final-postprocessing',
        'final-postprocessing-overlaps'
        :type classification_type: str
        :return: True in case the given name is avalid Sentinel2 image name
        :rtype: bool
    """
    assert isinstance(name, str), "Wrong input parameter for 'name', expected str"
    assert isinstance(classification_type, str), "Wrong input parameter for 'classification_type', expected str"

    regex_pattern = None
    if classification_type.lower() == 'trial':
        regex_pattern = configuration.CLASSIFICATION_OUTPUT_FILE_NAME
        regex_pattern = regex_pattern.replace("{}", "\\d+")
    elif classification_type.lower() == 'final-classified':
        regex_pattern = configuration.CLASSIFICATION_OUTPUT_FINAL_FILE_NAME
        regex_pattern = regex_pattern.replace("{}", ".+")
    elif classification_type.lower() == 'final-postprocessing':
        regex_pattern = configuration.POST_PROCESSING_FINAL_IMAGE_NAME
        regex_pattern = regex_pattern.replace("{}", ".+")
    elif classification_type.lower() == 'final-postprocessing-overlaps':
        regex_pattern = configuration.CLASSIFICATION_OVERLAP_MAJORITY_FILE_NAME
        regex_pattern = regex_pattern.replace("{}", ".+")
    else:
        raise SeomException("Unable to retrieve regular expression for the given 'type' " + type)

    file_name_date_regex = re.compile(regex_pattern)
    regex_matches = file_name_date_regex.match(name)

    return regex_matches is not None


def is_valid_sentinel2_image_name(name):
    """
        Check against a configured regular expression if the given name is a valid Sentinel2 image
        :param name: The candidate Sentinel2 image name
        :type name: str
        :return: True in case the given name is a valid Sentinel2 image name
        :rtype: bool
    """
    assert isinstance(name, str), "Wrong input parameter for 'name', expected str"

    file_name_date_regex = re.compile(configuration.IMAGES_S2_FILE_NAME_DATE_REGEX)
    regex_matches = file_name_date_regex.match(name)

    return regex_matches is not None


def is_valid_cmask_image_name(name):
    """
        Check against a configured regular expression if the given name is a valid cloud mask image
        :param name: The candidate cloud mask image name
        :type name: str
        :return: True in case the given name is a valid cloud mask image name
        :rtype: bool
    """
    assert isinstance(name, str), "Wrong input parameter for 'name', expected str"

    file_name_date_regex = re.compile(configuration.IMAGES_S2_CLOUD_MASK_FILE_NAME_DATE_REGEX)
    regex_matches = file_name_date_regex.match(name)

    return regex_matches is not None


def retrieve_image_matrix(image_path, target_type=numpy.float32, bands=None, logger=None):
    """
        Retrieve from the given path the image as a matrix
        :param image_path: The image path
        :type image_path: str
        :param target_type: The target type in the matrix
        :type target_type: numpy.number
        :param bands: The bands to retrieve, if 'None' all the bands will be retrieved. It is 0 based, so the first band
        has the identifier 0.
        :type bands: list of int
        :return: The matrix representation for the image in the path
        :rtype: numpy.ndarray
    """
    return retrieve_image_matrix_with_spatial_references(image_path, target_type, bands, logger)[0]


def retrieve_image_matrix_with_spatial_references(image_path, target_type=numpy.float32, bands=None, logger=None):
    """
        Retrieve from the given path the image as a matrix
        :param image_path: The image path
        :type image_path: str
        :param target_type: The target type in the matrix
        :type target_type: numpy.number
        :param bands: The bands to retrieve, if 'None' all the bands will be retrieved. It is 0 based, so the first band
        has the identifier 0.
        :type bands: list of int
        :return: The matrix representation for the image in the path, its geo_transformation, its projection
        :rtype: numpy.ndarray, tuple, str
    """
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        raise SeomException("Unable to retrieve the image from path: " + image_path)

    gdal_images = retrieve_images_gdal([image_path])
    if gdal_images is None or len(gdal_images) == 0:
        raise SeomException("Unable to retrieve the Sentinel2 samples image from path: " + image_path)

    image_matrix = transform_image_to_matrix(gdal_images[0], target_type=target_type, bands=bands)
    return image_matrix, gdal_images[0].GetGeoTransform(), gdal_images[0].GetProjection()


def retrieve_images_gdal(images_paths, logger=None):
    """
        Retrieve the images from the paths
        :param images_paths: The images paths to retrieve
        :type images_paths: list
        :return: The list of images (gdal)
        :rtype: list of gdal.Dataset
    """
    assert isinstance(images_paths, list), "Wrong input parameter type for 'images_paths', expected list"
    assert all(isinstance(element, str) for element in images_paths)\
        , "Wrong input parameter type for 'images_paths', expected to be a list of str"

    result = []
    for image_path in images_paths:
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            raise SeomException("The path " + image_path + " doesn't refer a file")

        if not is_feasible_image_file(image_path):
            raise SeomException("The path " + image_path + " doesn't refer to a feasible image")

        image = gdal.Open(image_path)
        result.append(image)

    return result


def retrieve_legend_colors(legend_path, logger=None):
    """
        Read from the legend file (expected to be a CSV generated by QGis) into a map.
        :param legend_path: The legend file path
        :type legend_path: str
        :return: The map for the value and its tuple of colors in the form (red, green, blue, alpha) having values among
        0 and 255 included.
        :rtype: dict
    """
    absolute_path = os.path.expanduser(legend_path)
    assert isinstance(absolute_path, str), "Wrong type for 'legend_path', expected str"
    assert os.path.exists(absolute_path) and os.path.isfile(absolute_path), \
        "The given path doesn't exist or is not a file"

    # The following represent the file structure as exported by QGis, where each row reports the values described
    # in the 'names' attribute.
    legend_type = {
        'names': ('clc_value', 'red', 'green', 'blue', 'alpha', 'clc_name'),
        'formats': ('i2', 'i2', 'i2', 'i2', 'i2', 'U255')
    }

    # Read the data skipping the first two rows
    legend = numpy.loadtxt(absolute_path, delimiter=",", dtype=legend_type, skiprows=2)

    # Convert in a map for ease of use
    result = dict()
    for element in legend:
        result[element[0]] = (element[1], element[2], element[3], element[4])

    return result


def save_matrix_as_geotiff(image_matrix, destination_file_path, reference_image, gdal_type, geo_transform=None,
                           projection=None, colors_maps_array=None, row_offset=0, column_offset=0, override_output=True):
    """
        Save the image as matrix into a geotiff, considering the reference_image for the geospatial reference.
        Alternatively to the 'reference_image', there could be given the parameters 'geo_transform' and 'projection'.
        :param image_matrix: The image as matrix to save
        :type image_matrix: numpy.ndarray
        :param destination_file_path: The target destination for the geotiff
        :type destination_file_path: str
        :param reference_image: The reference image for the geospatial information
        :type reference_image: gdal.Dataset
        :param gdal_type: the data type for the resulting image
        :type gdal_type: the gdal types
        :param geo_transform: The gdal.Dataset geo transformation data
        :type geo_transform: object
        :param projection: The gdal.Dataset projection
        :type projection: str
        :param colors_maps_array: The colors map to use for saving the raster. Expected an array of maps having as key the
        class value and as value a tuple like: (red, green, blue, alpha) having values among 0 and 255 included.
        Each map in the array address a specific band (positional ordered)
        :type colors_maps_array: list
        :param row_offset: The offset for saving the data in the X coordinate
        :type row_offset: int
        :param column_offset: The oftransforfset for saving the data in the Y coordinate
        :type column_offset: int
        :param override_output: True for overriding the existing output
        :type override_output: bool
    """
    assert isinstance(image_matrix, numpy.ndarray), "Wrong type for 'image_matrix', expected numpy.ndarray"
    assert isinstance(destination_file_path, str), "Wrong type for 'destination_file_path', expected str"
    if reference_image is not None:
        assert isinstance(reference_image, gdal.Dataset), "Wrong type for 'reference_image', expected gdal.Dataset"
    else:
        assert isinstance(geo_transform, tuple), "Wrong type for 'geo_transform', expected tuple"
        assert isinstance(projection, str), "Wrong type for 'projection', expected str"
    if colors_maps_array is not None:
        assert isinstance(colors_maps_array, list), "Wrong type for 'colors_maps_array', expected list"
        assert all(isinstance(colors_map, dict) for colors_map in colors_maps_array), \
            "Wrong type for 'colors_maps_array', expected list of maps"
        assert all(all(len(colors) == 4 for colors in colors_map.values()) for colors_map in colors_maps_array), \
            "Wrong type for 'colors_maps_array' map values, expected tuple of (red, green, blue, alpha)"
        assert all(
            all(all(0 <= color <= 255 for color in colors) for colors in colors_map.values())
            for colors_map in colors_maps_array), \
            "Wrong type for 'colors_maps_array' map values, expected tuple of (red, green, blue, alpha) having " \
            "values among 0 and 255 included."
    assert isinstance(row_offset, int), "Wrong type for 'x_offset', expected int"
    assert isinstance(column_offset, int), "Wrong type for 'y_offset', expected int"

    destination_folder = os.path.dirname(destination_file_path)
    if not os.path.exists(destination_folder) and not os.path.isdir(destination_folder):
        raise SeomException("The destination file path folder doesn't exist")

    if reference_image is not None:
        geo_transform = reference_image.GetGeoTransform()
        projection = reference_image.GetProjection()
    else:
        geo_transform = geo_transform
        projection = projection

    driver = gdal.GetDriverByName('GTiff')

    bands = 1
    if len(image_matrix.shape) > 2:
        bands = image_matrix.shape[2]

    image_result = None
    existing = os.path.exists(destination_file_path) and os.path.isfile(destination_file_path)
    if not existing or (existing and override_output):
        # Creating the image
        image_result = driver.Create(
            destination_file_path,
            image_matrix.shape[1],
            image_matrix.shape[0],
            bands,
            gdal_type)
        image_result.SetGeoTransform(geo_transform)
        image_result.SetProjection(projection)
    else:
        # Acquiring the image for updating
        image_result = gdal.Open(destination_file_path, gdal.GA_Update)

    for band in range(0, bands):
        raster_band = image_result.GetRasterBand(band + 1)
        if len(image_matrix.shape) > 2:
            raster_band.WriteArray(image_matrix[:, :, band], xoff=column_offset, yoff=row_offset)
        else:
            raster_band.WriteArray(image_matrix[:, :], xoff=column_offset, yoff=row_offset)

        # In case the colors map has not been given or there aren't enough elements in it, skip
        if colors_maps_array is None or len(colors_maps_array) < band + 1:
            continue

        # Prepare the color map for the band
        colors_map = colors_maps_array[band]
        colors_table = gdal.ColorTable()
        for class_value in colors_map.keys():
            # Note that the alpha channel seems not supported by the TIFF format, as described here:
            # https://gdal.org/drivers/raster/gtiff.html#creation-issues
            colors_table.SetColorEntry(int(class_value), colors_map[class_value])

        # Setting the color map to the band
        raster_band.SetRasterColorTable(colors_table)

    image_result.FlushCache()


def transform_image_to_matrix(image, target_type, bands=None, logger=None):
    """
        Transform the given image into a matrix
        :param image: The input image to transform
        :type image: gdal.Dataset
        :param target_type: The target type in the matrix
        :type target_type: numpy.number
        :param bands: The bands to retrieve, if 'None' all the bands will be retrieved. It is 0 based, so the first band
        has the identifier 0.
        :type bands: list of int
        :return: The resulting matrix. It will be a x * y * z (even when just a band ~ plain image)
        :rtype: numpy.ndarray
    """
    assert isinstance(image, gdal.Dataset), "Wrong input parameter type for 'image', expected gdal.Dataset"
    assert numpy.issubdtype(target_type, numpy.number)\
        , "Wrong input parameter type for 'target_type', expected numpy.number"

    bands_count = image.RasterCount
    x_size = image.RasterYSize
    y_size = image.RasterXSize

    result_bands_size = bands_count
    if bands is not None and len(bands) > 0:
        result_bands_size = len(bands)

    # Preparing the matrix which will host the result
    result = numpy.zeros((x_size, y_size, result_bands_size), dtype=target_type)

    # Iterate over all the bands
    result_band = 0
    for i in range(0, bands_count):
        # Skip the band if a selection of them has been requested to load
        if bands is not None and i not in bands:
            continue

        # Acquire the band
        band = image.GetRasterBand(i + 1).ReadAsArray().astype(target_type)

        # Update the image resulting matrix for the band, normalizing the values
        result[:, :, result_band] = band
        result_band += 1

    return result


def retrieve_s2_images_from_remote(application_context, temporary_path, remote_server_password, logger, k=None,
                                   skip_loaded=True):
    """
        Load the Sentinel2 images from remote into memory.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param temporary_path: The temporary path for downloading the resource
        :type temporary_path: str
        :param remote_server_password: The remote server password for the configured account
        :type remote_server_password: str
        :param logger: The logger to use
        :type logger: logging.Logger
        :param k: The top-k number of images to retrieve (alphabetically sorted)
        :type k: int
        :param skip_loaded: True for skipping those images which are already in the given Application Context
        :type skip_loaded: bool
        :return: The Sentinel2 images
        :rtype: Sentinel2Images
    """
    # TODO use application_context logger
    # TODO use application_context temporary path
    # TODO use application_context remote password
    # TODO delegate on retrieve_image_matrix_from_remote
    # TODO move to remote_operations
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"
    assert isinstance(temporary_path, str), \
        "Wrong input parameter type for 'temporary_path', expected str"
    assert os.path.exists(temporary_path), "The path {} doesn't exist".format(temporary_path)
    assert isinstance(remote_server_password, str), \
        "Wrong input parameter type for 'remote_server_password', expected str"
    assert isinstance(logger, logging.Logger), "Wrong input parameter type for 'logger', expected logging.Logger"
    assert isinstance(skip_loaded, bool), "Wrong input parameter type for 'skip_loaded', expected bool"
    if k is not None:
        assert isinstance(k, int), "Wrong input parameter type for 'k', expected int"

    # The result
    sentinel2_images = None

    transport = paramiko.Transport((configuration.REMOTE_HOST, 22))
    try:
        transport.connect(username=configuration.REMOTE_USERNAME, password=remote_server_password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(None)

        # Getting the elements in the remote position filtered by valid Sentinel 2 name
        remote_elements = sorted(sftp.listdir(application_context.input_parameters.s2_data_path))
        remote_elements = [element for element in remote_elements if is_valid_sentinel2_image_name(element)]

        if k is not None:
            if len(remote_elements) < k:
                raise SeomException("There are fewer elements available than the requested ones")

            remote_elements = remote_elements[:k]

        temporary_folder_path = get_temporary_folder_path(application_context, temporary_path)
        for remote_element in remote_elements:
            if not is_valid_sentinel2_image_name(remote_element):
                continue

            if application_context.s2_images is not None and application_context.s2_images.name_matrix_map is not None \
                and remote_element in application_context.s2_images.name_matrix_map \
                    and application_context.s2_images.name_matrix_map[remote_element] is not None:
                logger.info("The {} already loaded in memory".format(remote_element))
                continue

            # Download the image
            remote_path = os.path.join(application_context.input_parameters.s2_data_path, remote_element)
            local_path = os.path.join(temporary_folder_path, remote_element)
            if not configuration.OVERWRITE_LOCAL and os.path.exists(local_path):
                logger.info("The file {} already exists, not required to overwrite".format(local_path))
            else:
                logger.info("Downloading {}".format(remote_path))
                sftp.get(remote_path, local_path)

            # Load the image
            s2_matrix, geo_transform, projection = retrieve_image_matrix_with_spatial_references(
                local_path, target_type=configuration.IMAGES_S2_NUMPY_TYPE)
            if sentinel2_images is None:
                sentinel2_images = Sentinel2Images()
                sentinel2_images.path = 'in-memory'
                sentinel2_images.projection = projection
                sentinel2_images.geo_transform = geo_transform
                sentinel2_images.x_size = s2_matrix.shape[0]
                sentinel2_images.y_size = s2_matrix.shape[1]
                sentinel2_images.bands_size = s2_matrix.shape[2]

            # Load the image into memory
            sentinel2_images.name_matrix_map[remote_element] = s2_matrix

            # Delete the image from file system
            os.remove(local_path)
    except Exception as e:
        logger.error("Unable to process the Sentinel2 images due arisen error")
        logger.critical(traceback.format_exc())
        raise SeomException("Unable to retrieve data")
    finally:
        transport.close()

    return sentinel2_images


def retrieve_classified_images_from_remote(application_context, temporary_path, remote_server_password, logger):
    """
        Load the classified images from remote into memory.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param temporary_path: The temporary path for downloading the resource
        :type temporary_path: str
        :param remote_server_password: The remote server password for the configured account
        :type remote_server_password: str
        :param logger: The logger to use
        :type logger: logging.Logger
        :return: The predicted_images, predicted_image, geo_transformation, projection
        :rtype: Tuple list, numyp.ndarray, object, object
    """
    # TODO use application_context logger
    # TODO use application_context temporary path
    # TODO use application_context remote password
    # TODO delegate on retrieve_image_matrix_from_remote
    # TODO move to remote_operations

    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context' expected 'ApplicationContext'"
    assert isinstance(temporary_path, str), \
        "Wrong input parameter type for 'temporary_path', expected str"
    assert os.path.exists(temporary_path), "The path {} doesn't exist".format(temporary_path)
    assert isinstance(remote_server_password, str), \
        "Wrong input parameter type for 'remote_server_password', expected str"
    assert isinstance(logger, logging.Logger), "Wrong input parameter type for 'logger', expected logging.Logger"
    assert application_context.tile_year > 0, "The application context doesn't have a valid year set"
    assert isinstance(application_context.tile_name, str), "The application context doesn't have a valid tile name"

    # Target remote path
    remote_path = get_remote_output_path(application_context)

    # Response
    predicted_images = []
    predicted_image = None
    geo_transformation = None
    projection = None

    transport = paramiko.Transport((configuration.REMOTE_HOST, 22))
    try:
        transport.connect(username=configuration.REMOTE_USERNAME, password=remote_server_password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(None)

        # Getting the elements in the remote position filtered by feasible extension
        logger.debug("Retrieving list of files from " + remote_path)
        remote_elements = sorted(sftp.listdir(remote_path))
        remote_elements = [element for element in remote_elements
                           if str(os.path.splitext(element)[1]).lower() in configuration.IMAGES_FEASIBLE_EXTENSIONS and
                           (is_valid_classified_image_name(element, 'trial') or
                            is_valid_classified_image_name(element, 'final'))
                           ]

        # temporary_folder_path = get_temporary_folder_path(application_context, temporary_path)
        for remote_element in remote_elements:
            # Download the image
            remote_path = os.path.join(remote_path, remote_element)
            local_path = os.path.join(temporary_path, remote_element)
            if not configuration.OVERWRITE_LOCAL and os.path.exists(local_path):
                logger.info("The file {} already exists, not required to overwrite".format(local_path))
            else:
                logger.info("Downloading {}".format(remote_path))
                sftp.get(remote_path, local_path)

            # Load the image into memory
            image_matrix, geo_transformation, projection = \
                retrieve_image_matrix_with_spatial_references(local_path,
                                                              target_type=configuration.IMAGES_PREDICTED_NUMPY_TYPE)

            if is_valid_classified_image_name(remote_element, 'trial'):
                predicted_images.append(image_matrix)
            else:
                predicted_image = image_matrix

            # Delete the image from file system
            os.remove(local_path)
    except Exception as e:
        logger.error("Unable to process the classified images due arisen error")
        logger.critical(traceback.format_exc())
        raise SeomException("Unable to retrieve data")
    finally:
        transport.close()

    return predicted_images, predicted_image, geo_transformation, projection


def retrieve_s2_cloud_masks_images_from_remote(application_context, temporary_path, remote_server_password,
                                               logger):
    """
        Load the cloud masks images corresponding to the loaded Sentinel2 images in the application context from remote
        into memory.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param temporary_path: The temporary path for downloading the resource
        :type temporary_path: str
        :param remote_server_password: The remote server password for the configured account
        :type remote_server_password: str
        :param logger: The logger to use
        :type logger: logging.Logger
        :return: The map having keys the Sentinel2 image name and as value the corresponding cloud mask
        :rtype: dict
    """
    # TODO use application_context logger
    # TODO use application_context temporary path
    # TODO use application_context remote password
    # TODO delegate on retrieve_image_matrix_from_remote
    # TODO move to remote_operations
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected 'ApplicationContext'"
    assert isinstance(application_context.input_parameters, InputParameters), \
        "Wrong input parameter type for 'application_context.input_parameters', expected 'InputParameters'"
    assert isinstance(application_context.input_parameters.cloud_mask_image_path, str), \
        "Wrong input parameter type for 'application_context.input_parameters.cloud_mask_image_path', expected str"
    assert isinstance(application_context.s2_images, Sentinel2Images), \
        "Wrong input parameter type for 'application_context.s2_images', expected 'Sentinel2Images'"
    assert isinstance(application_context.s2_images.name_matrix_map, dict), \
        "Wrong input parameter type for 'application_context.s2_images.name_matrix_map', expected dict"
    assert isinstance(temporary_path, str), \
        "Wrong input parameter type for 'temporary_path', expected str"
    assert os.path.exists(temporary_path), "The path {} doesn't exist".format(temporary_path)
    assert isinstance(remote_server_password, str), \
        "Wrong input parameter type for 'remote_server_password', expected str"
    assert isinstance(logger, logging.Logger), "Wrong input parameter type for 'logger', expected logging.Logger"

    # Retrieving the Sentinel2 cloud masks
    s2_name_cloud_mask_path_map = get_s2_cloud_masks_images_file_paths_map(application_context)

    # The result
    cloud_masks_images = dict()

    transport = paramiko.Transport((configuration.REMOTE_HOST, 22))
    try:
        transport.connect(username=configuration.REMOTE_USERNAME, password=remote_server_password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(None)
        temporary_folder_path = get_temporary_folder_path(application_context, temporary_path)

        for s2_image_name in s2_name_cloud_mask_path_map:
            cloud_mask_image_path = s2_name_cloud_mask_path_map[s2_image_name]
            # Download the image
            cloud_mask_file_name = os.path.basename(cloud_mask_image_path)
            local_path = os.path.join(temporary_folder_path, cloud_mask_file_name)
            if not configuration.OVERWRITE_LOCAL and os.path.exists(local_path):
                logger.info("The file {} already exists, not required to overwrite".format(local_path))
            else:
                logger.info("Downloading {}".format(cloud_mask_image_path))
                sftp.get(cloud_mask_image_path, local_path)

            # Load the image into memory
            cloud_mask_image = retrieve_image_matrix(local_path, target_type=configuration.IMAGES_S2_NUMPY_TYPE)
            cloud_masks_images[s2_image_name] = cloud_mask_image

            # Delete the image from file system
            os.remove(local_path)
    except Exception as e:
        logger.error("Unable to process the cloud masks images due arisen error")
        logger.critical(traceback.format_exc())
        raise SeomException("Unable to retrieve data")
    finally:
        transport.close()

    return cloud_masks_images


def retrieve_clc_image_from_remote(application_context, temporary_path, remote_server_password, logger):
    """
        Load the CLC (Corine Land Cover) image from remote into memory.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param temporary_path: The temporary path for downloading the resource
        :type temporary_path: str
        :param remote_server_password: The remote server password for the configured account
        :type remote_server_password: str
        :param logger: The logger to use
        :type logger: logging.Logger
        :return: The CLC image
        :rtype: numpy.ndarray
    """
    # TODO use application_context logger
    # TODO use application_context temporary path
    # TODO use application_context remote password
    # TODO delegate on retrieve_image_matrix_from_remote
    # TODO move to remote_operations
    assert isinstance(application_context,
                      ApplicationContext), "Wrong input parameter type, expected 'ApplicationContext'"
    assert isinstance(temporary_path, str), \
        "Wrong input parameter type for 'temporary_path', expected str"
    assert os.path.exists(temporary_path), "The path {} doesn't exist".format(temporary_path)
    assert isinstance(remote_server_password, str), \
        "Wrong input parameter type for 'remote_server_password', expected str"
    assert isinstance(logger, logging.Logger), "Wrong input parameter type for 'logger', expected logging.Logger"

    # The result
    clc_image = None

    transport = paramiko.Transport((configuration.REMOTE_HOST, 22))
    try:
        transport.connect(username=configuration.REMOTE_USERNAME, password=remote_server_password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(None)
        temporary_folder_path = get_temporary_folder_path(application_context, temporary_path)

        # Download the image
        file_name = os.path.basename(application_context.input_parameters.corine_land_cover_data_path)
        local_path = os.path.join(temporary_folder_path, file_name)
        if not configuration.OVERWRITE_LOCAL and os.path.exists(local_path):
            logger.info("The file {} already exists, not required to overwrite".format(local_path))
        else:
            logger.info("Downloading {}".format(application_context.input_parameters.corine_land_cover_data_path))
            sftp.get(application_context.input_parameters.corine_land_cover_data_path, local_path)

        # Load the image into memory
        clc_image = retrieve_image_matrix(local_path, target_type=configuration.IMAGES_CLC_NUMPY_TYPE)

        # Delete the image from file system
        os.remove(local_path)
    except Exception as e:
        logger.error("Unable to process the CLC image due arisen error")
        logger.critical(traceback.format_exc())
        raise SeomException("Unable to retrieve data")
    finally:
        transport.close()

    return clc_image


def retrieve_cloud_mask_image_from_remote(application_context, temporary_path, remote_server_password, logger):
    """
        Load the cloud mask image from remote into memory.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param temporary_path: The temporary path for downloading the resource
        :type temporary_path: str
        :param remote_server_password: The remote server password for the configured account
        :type remote_server_password: str
        :param logger: The logger to use
        :type logger: logging.Logger
        :return: The cloud mask image
        :rtype: numpy.ndarray
    """
    # TODO use application_context logger
    # TODO use application_context temporary path
    # TODO use application_context remote password
    # TODO delgate on retrieve_image_matrix_from_remote
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected 'ApplicationContext'"
    assert isinstance(temporary_path, str), \
        "Wrong input parameter type for 'temporary_path', expected str"
    assert os.path.exists(temporary_path), "The path {} doesn't exist".format(temporary_path)
    assert isinstance(remote_server_password, str), \
        "Wrong input parameter type for 'remote_server_password', expected str"
    assert isinstance(logger, logging.Logger), "Wrong input parameter type for 'logger', expected logging.Logger"

    # The result
    cloud_mask_image = None

    transport = paramiko.Transport((configuration.REMOTE_HOST, 22))
    try:
        transport.connect(username=configuration.REMOTE_USERNAME, password=remote_server_password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(None)
        temporary_folder_path = get_temporary_folder_path(application_context, temporary_path)

        # Download the image
        cloud_mask_name = os.path.basename(application_context.input_parameters.cloud_mask_image_path)
        local_path = os.path.join(temporary_folder_path, cloud_mask_name)
        if not configuration.OVERWRITE_LOCAL and os.path.exists(local_path):
            logger.info("The file {} already exists, not required to overwrite".format(local_path))
        else:
            logger.info("Downloading {}".format(application_context.input_parameters.cloud_mask_image_path))
            sftp.get(application_context.input_parameters.cloud_mask_image_path, local_path)

        # Load the image into memory
        cloud_mask_image = retrieve_image_matrix(local_path, target_type=configuration.IMAGES_S2_NUMPY_TYPE)

        # Delete the image from file system
        os.remove(local_path)
    except Exception as e:
        logger.error("Unable to process the cloud mask image due arisen error")
        logger.critical(traceback.format_exc())
        raise SeomException("Unable to retrieve data")
    finally:
        transport.close()

    return cloud_mask_image


def retrieve_cloud_mask_images_from_remote(application_context, temporary_path, remote_server_password, logger, k=None):
    """
        Load the cloud mask images from remote into memory, each corresponding to the Sentinel2 Images.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param temporary_path: The temporary path for downloading the resource
        :type temporary_path: str
        :param remote_server_password: The remote server password for the configured account
        :type remote_server_password: str
        :param logger: The logger to use
        :type logger: logging.Logger
        :param k: The top-k number of images to retrieve (alphabetically sorted)
        :type k: int
        :return: The cloud mask images dictionary
        :rtype: dict having as key the corresponding sentintel2 image name and as value the numpy.ndarray image
    """
    # TODO use application_context logger
    # TODO use application_context temporary path
    # TODO use application_context remote password
    # TODO delgate on retrieve_image_matrix_from_remote
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected 'ApplicationContext'"
    assert isinstance(temporary_path, str), \
        "Wrong input parameter type for 'temporary_path', expected str"
    assert os.path.exists(temporary_path), "The path {} doesn't exist".format(temporary_path)
    assert isinstance(remote_server_password, str), \
        "Wrong input parameter type for 'remote_server_password', expected str"
    assert isinstance(logger, logging.Logger), "Wrong input parameter type for 'logger', expected logging.Logger"
    if k is not None:
        assert isinstance(k, int), "Wrong input parameter type for 'k', expected int"

    # The result
    cloud_mask_images = dict()

    transport = paramiko.Transport((configuration.REMOTE_HOST, 22))
    try:
        transport.connect(username=configuration.REMOTE_USERNAME, password=remote_server_password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(None)

        # The remote path the cmasks are stored
        remote_folder_path = os.path.dirname(application_context.input_parameters.cloud_mask_image_path)

        # Getting the elements in the remote position filtered by valid Sentinel 2 name
        remote_elements = sorted(sftp.listdir(remote_folder_path))
        remote_elements = [element for element in remote_elements if is_valid_cmask_image_name(element)]

        # Cloud mask name to Sentinel 2 name map
        cmask_to_s2_map = dict()

        # The remote elements to download
        remote_elements_to_download = []
        for s2_name in application_context.s2_images.name_matrix_map:
            year, month, day = get_date_components_by_text(s2_name)
            date = str(year) + str(month).zfill(2) + str(day).zfill(2)

            candidates = [element for element in remote_elements if date in element]
            if len(candidates) == 0:
                continue

            if len(candidates) > 1:
                raise SeomException("There are multiple files for date " + str(date) + " on " + remote_folder_path)

            remote_elements_to_download.append(candidates[0])
            cmask_to_s2_map[candidates[0]] = s2_name

        if k is not None:
            if len(remote_elements_to_download) < k:
                raise SeomException("There are fewer elements available than the requested ones")

            remote_elements_to_download = remote_elements_to_download[:k]

        # Downloading data
        temporary_folder_path = get_temporary_folder_path(application_context, temporary_path)
        for element_name in remote_elements_to_download:
            # Download the image
            remote_path = os.path.join(remote_folder_path, element_name)
            local_path = os.path.join(temporary_folder_path, element_name)
            if not configuration.OVERWRITE_LOCAL and os.path.exists(local_path):
                logger.info("The file {} already exists, not required to overwrite".format(local_path))
            else:
                logger.info("Downloading {}".format(remote_path))
                sftp.get(remote_path, local_path)

            # Load the image
            cloud_matrix = retrieve_image_matrix(local_path, target_type=configuration.IMAGES_S2_NUMPY_TYPE)

            # Load the image into memory
            cloud_mask_images[cmask_to_s2_map[element_name]] = cloud_matrix

            # Delete the image from file system
            os.remove(local_path)

        return cloud_mask_images
    except Exception as e:
        logger.error("Unable to process the cloud mask image due arisen error")
        logger.critical(traceback.format_exc())
        raise SeomException("Unable to retrieve data")
    finally:
        transport.close()

    return None


def retrieve_image_matrix_from_remote(target_file_path, temporary_folder_path, remote_server_password, logger,
                                      target_type=configuration.IMAGES_S2_NUMPY_TYPE):
    """
        Load the image from remote into memory.
        :param target_file_path: The path from which retrieve the image
        :type target_file_path: str
        :param temporary_folder_path: The temporary path for downloading the resource
        :type temporary_folder_path: str
        :param remote_server_password: The remote server password for the configured account
        :type remote_server_password: str
        :param logger: The logger to use
        :type logger: logging.Logger
        :param target_type: The target type for the acquired matrix image
        :type target_type: numpy.number
        :return: The image, the geo_transformation, the projection
        :rtype: numpy.ndarray, tuple, str
    """
    # TODO use application_context logger
    # TODO use application_context temporary path
    # TODO use application_context remote password
    assert isinstance(target_file_path, str), \
        "Wrong input parameter type for 'target_path', expected str"
    assert isinstance(temporary_folder_path, str), \
        "Wrong input parameter type for 'temporary_folder_path', expected str"
    assert os.path.exists(temporary_folder_path), "The path {} doesn't exist".format(temporary_folder_path)
    assert isinstance(remote_server_password, str), \
        "Wrong input parameter type for 'remote_server_password', expected str"
    assert isinstance(logger, logging.Logger), "Wrong input parameter type for 'logger', expected logging.Logger"

    # The result
    image_matrix = None
    geo_transform = None
    projection = None

    transport = paramiko.Transport((configuration.REMOTE_HOST, 22))
    try:
        transport.connect(username=configuration.REMOTE_USERNAME, password=remote_server_password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(None)

        # Download the image
        target_name = os.path.basename(target_file_path)
        local_path = os.path.join(temporary_folder_path, target_name)
        if not configuration.OVERWRITE_LOCAL and os.path.exists(local_path):
            logger.info("The file {} already exists, not required to overwrite".format(local_path))
        else:
            logger.info("Downloading {}".format(target_file_path))
            sftp.get(target_file_path, local_path)

        # Load the image into memory
        image_matrix, geo_transform, projection = retrieve_image_matrix(local_path, target_type=target_type)

        # Delete the image from file system
        os.remove(local_path)
    except Exception as e:
        logger.error("Unable to process the image path " + target_file_path + " due arisen error")
        logger.critical(traceback.format_exc())
        raise SeomException("Unable to retrieve data")
    finally:
        transport.close()

    return image_matrix, geo_transform, projection
