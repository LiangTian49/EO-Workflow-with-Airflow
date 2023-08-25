#!/usr/bin/env python
import logging
import os
import traceback

import gc
import numpy
from sklearn.cluster import KMeans

from configurations import configuration
from modules import utils, spectral_indices, logging_utils
from modules.exceptions.SeomException import SeomException
from modules.models.application_context import ApplicationContext
from modules.models.sentinel2_images import Sentinel2Images


def validate_step_data(application_context):
    """
        Validates all the data required by the step. Not all the given content will be validated, just the ones for
        the current step
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    assert isinstance(application_context, ApplicationContext),\
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    assert application_context.s2_images is not None, "The 's2_images' is not set"
    assert application_context.clc_map_converted_matrix is not None, "The 'clc_map_converted_matrix' is not set"

    assert isinstance(application_context.s2_images, Sentinel2Images),\
        "Wrong type for 's2_images', expected Sentinel2Images"
    assert isinstance(application_context.clc_map_converted_matrix, numpy.ndarray),\
        "Wrong type for 'clc_map_converted_matrix', expected numpy.ndarray"

    if not configuration.SOURCE_REMOTE:
        assert os.path.exists(application_context.s2_images.path) and \
               os.path.isdir(application_context.s2_images.path), \
               "The path for Sentinel2 images doesn't exist or it is not a folder"
    assert isinstance(application_context.input_parameters.s2_samples_image_name, str) and\
        application_context.input_parameters.s2_samples_image_name is not None and\
        len(application_context.input_parameters.s2_samples_image_name) > 0,\
        "The input parameter for the Sentinel2 image for computing the samples is expected to be a string representing " \
        "a file name"


def get_indexes_matrix(matrix, logger):
    """
        Computing the indexes for the matrix.
        :param matrix: The matrix to use for composing the indexes matrix. Expected to be a 3D matrix (representing a
        multispectral image)
        :type matrix: numpy.ndarray
        :param logger: The logger
        :type logger: logging.Logger
        :return: The computed indexes for the given matrix. The 3D dimension will have the indexes in the order:
        NDVI, NDWI, NDSI, EVI, RockAndSand
        :rtype: numpy.ndarray
    """
    assert isinstance(matrix, numpy.ndarray),\
        "Wrong type for 'matrix', expected numpy.ndarray"

    # For all reshaping: using order 'F' (Fortran ~ column major) for keeping consistency with MATLAB
    logger.debug("Computing NDVI")
    ndvi_matrix = spectral_indices.get_ndvi_matrix(matrix)
    ndvi_matrix = numpy.reshape(ndvi_matrix, ndvi_matrix.shape[0] * ndvi_matrix.shape[1], order='F')

    logger.debug("Computing NDWI")
    ndwi_matrix = spectral_indices.get_ndwi_matrix(matrix)
    ndwi_matrix = numpy.reshape(ndwi_matrix, ndwi_matrix.shape[0] * ndwi_matrix.shape[1], order='F')

    # Processing the resulting feature space at each step so to keep the memory usage lower
    indexes_matrix = numpy.column_stack([ndvi_matrix, ndwi_matrix])
    ndvi_matrix = None
    ndwi_matrix = None

    logger.debug("Computing NDSI")
    ndsi_matrix = spectral_indices.get_ndsi_matrix(matrix)
    ndsi_matrix = numpy.reshape(ndsi_matrix, ndsi_matrix.shape[0] * ndsi_matrix.shape[1], order='F')
    indexes_matrix = numpy.column_stack([indexes_matrix, ndsi_matrix])
    ndsi_matrix = None

    logger.debug("Computing EVI")
    evi_matrix = spectral_indices.get_evi_matrix(matrix)
    evi_matrix = numpy.reshape(evi_matrix, evi_matrix.shape[0] * evi_matrix.shape[1], order='F')
    indexes_matrix = numpy.column_stack([indexes_matrix, evi_matrix])
    evi_matrix = None

    logger.debug("Computing rock sand index")
    rock_sand_matrix = spectral_indices.get_rock_sand_index_matrix(matrix)
    rock_sand_matrix = numpy.reshape(rock_sand_matrix, rock_sand_matrix.shape[0] * rock_sand_matrix.shape[1], order='F')
    indexes_matrix = numpy.column_stack([indexes_matrix, rock_sand_matrix])
    rock_sand_matrix = None

    return indexes_matrix


def get_samples_matrix(application_context):
    """
        Retrieve the sample image (from Sentinel2 data path)
        :param application_context: The application context
        :type application_context: ApplicationContext
        :return: The samples image
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext),\
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    # In case it is already in memory, just retrive it
    s2_images = application_context.s2_images
    if s2_images is not None and \
        s2_images.name_matrix_map is not None and \
        application_context.input_parameters.s2_samples_image_name in s2_images.name_matrix_map and \
        s2_images.name_matrix_map[application_context.input_parameters.s2_samples_image_name] is not None:
        application_context.logger.info("The image is already in memory")
        return application_context.s2_images.name_matrix_map[application_context.input_parameters.s2_samples_image_name]

    try:
        path = os.path.join(application_context.s2_images.path,
                            application_context.input_parameters.s2_samples_image_name)
        application_context.logger.info("Reading the sample image " + path)

        samples_matrix = utils.retrieve_image_matrix(path, target_type=configuration.IMAGES_S2_NUMPY_TYPE)
        return samples_matrix
    except SeomException as e:
        raise
    except Exception as e:
        raise SeomException("Unable to retrieve the Sentinel2 samples image", e)


def get_cloud_free_clc_matrix(application_context):
    """
        Retrieve the Corine Land Cover (CLC) matrix without the area being covered by clouds in the Sentinel2 samples
        image
        :param application_context: The application context
        :type application_context: ApplicationContext
        :return: The samples image
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext),\
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    cloud_cover_mask_matrix = None
    if application_context.cloud_mask_image is not None:
        application_context.logger.info("The cloud mask image is already in memory")
        cloud_cover_mask_matrix = application_context.cloud_mask_image
    else:
        try:
            path = application_context.input_parameters.cloud_mask_image_path
            application_context.logger.info("Reading the cloud cover image " + path)
            if not os.path.exists(path) or not os.path.isfile(path):
                raise SeomException("Unable to retrieve the cloud cover mask image from path: " + path)

            cloud_cover_mask_image = utils.retrieve_images_gdal([path])
            if cloud_cover_mask_image is None or len(cloud_cover_mask_image) == 0:
                raise SeomException("Unable to retrieve the cloud cover mask image from path: " + path)

            cloud_cover_mask_matrix = utils.transform_image_to_matrix(
                cloud_cover_mask_image[0],
                target_type=configuration.IMAGES_CLC_NUMPY_TYPE)

            # For freeing memory
            cloud_cover_mask_image = None
        except SeomException as e:
            raise
        except Exception as e:
            raise SeomException("Unable to retrieve the Sentinel2 samples image", e)

    # Removing the cloud areas
    application_context.logger.info("Removing the areas of the CLC which have clouds in the Sentinel2 image")
    clc_map_converted_matrix_cloud_free = application_context.clc_map_converted_matrix.copy()
    clc_map_converted_matrix_cloud_free[cloud_cover_mask_matrix == 1] = 0
    clc_map_converted_matrix_cloud_free = numpy.reshape(
        clc_map_converted_matrix_cloud_free,
        (clc_map_converted_matrix_cloud_free.shape[0] * clc_map_converted_matrix_cloud_free.shape[1], 1),
        order='F')
    return clc_map_converted_matrix_cloud_free


def generate_samples(
        application_context,
        indexes_matrix_normalized,
        clc_indexes_for_class,
        class_value,
        clusters_for_class,
        clusters_to_keep_for_class
        ):
    """
        Perform the samples selection
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param indexes_matrix_normalized: The matrix of the indexes (e.g. NDVI, NDWI...) computed on the samples
        Sentinel2 image
        :type indexes_matrix_normalized: numpy.ndarray
        :param clc_indexes_for_class: The Corine Land Cover (cloud free) map indexes for the representing class
        :type clc_indexes_for_class: numpy.ndarray
        :param class_value: The value for the class (e.g. 1)
        :type class_value: int
        :param clusters_for_class: The number of clusters to look for in the given matrix
        :type clusters_for_class: int
        :param clusters_to_keep_for_class: The number of clusters to keep in the result
        :type clusters_to_keep_for_class: int
        :return: List of samples from the given matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext),\
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(clc_indexes_for_class, numpy.ndarray), "Wrong type for 'mask_for_class', expected numpy.ndarray"

    # The array of all the pixels marked in clusters
    all_pixels = numpy.array([])
    # The dictionary having the key the cluster value (will be 1..n) with the associated cluster percentage
    clusters_statistics = dict()

    if clc_indexes_for_class.shape[0] >= clusters_for_class:
        kmeans_model = KMeans(
            n_clusters=clusters_for_class,
            random_state=configuration.SAMPLES_KMEANS_RANDOM_STATE,
            max_iter=configuration.SAMPLES_KMEANS_MAX_ITERATIONS,
            n_jobs=utils.get_processors_count_classification()) \
            .fit(indexes_matrix_normalized[clc_indexes_for_class])
        clustering_results = kmeans_model.labels_

        # Some stats
        for cluster_identifier in range(0, clusters_for_class):
            cluster_elements = clustering_results[clustering_results == cluster_identifier]
            cluster_percentage = (cluster_elements.shape[0] / clustering_results.shape[0]) * 100
            clusters_statistics[cluster_identifier] = cluster_percentage

        # Sorting by percentage
        clusters_statistics = sorted(clusters_statistics.items(), key=lambda key_value: key_value[1], reverse=True)

        # Computing the distances from the cluster centers
        transformed = kmeans_model.transform(indexes_matrix_normalized[clc_indexes_for_class])

        # Computing the quantile for each cluster
        target = configuration.SAMPLES_QUANTILE_THRESHOLD
        for cluster_statistic_tuple in clusters_statistics:
            cluster_value = cluster_statistic_tuple[0]

            # In case the cluster doesn't have any elements, skip he quantization
            if transformed[clustering_results == cluster_value, cluster_value].size == 0:
                continue

            # Computing the threshold representing the quantile
            cluster_threshold = numpy.quantile(transformed[clustering_results == cluster_value, cluster_value], target)

            # Removing uncertain pixels (setting negative because KMeans uses clusters from 0..n)
            clustering_results[numpy.logical_and(
                clustering_results == cluster_value,
                transformed[:, cluster_value] > cluster_threshold)
            ] = -1

        all_pixels = numpy.stack((
            numpy.ones(clc_indexes_for_class.shape[0], dtype=configuration.IMAGES_CLC_NUMPY_TYPE) * class_value,
            clc_indexes_for_class,
            clustering_results
        ),
            axis=1)
    else:
        # Required for generating a file anyway
        all_pixels = numpy.stack((
            numpy.ones(clc_indexes_for_class.shape[0], dtype=configuration.IMAGES_CLC_NUMPY_TYPE) * class_value,
            clc_indexes_for_class,
            numpy.zeros(clc_indexes_for_class.shape[0], dtype=configuration.IMAGES_CLC_NUMPY_TYPE)
        ),
            axis=1)

    # The array of the selected pixels to keep
    selected_pixels = numpy.array([])

    if all_pixels is None or len(all_pixels) == 0 or all_pixels.shape[0] == 0:
        return selected_pixels

    # Compose the result by considering the number of clusters to keep
    clustering_results_filter = None
    for i in range(0, clusters_to_keep_for_class):
        cluster_statistic_tuple = clusters_statistics[i]
        cluster_value = cluster_statistic_tuple[0]
        if clustering_results_filter is None:
            clustering_results_filter = all_pixels[:, 2] == cluster_value
            continue

        clustering_results_filter = numpy.logical_or(clustering_results_filter, all_pixels[:, 2] == cluster_value)

    # Getting the pixels accordingly with the filter
    selected_pixels = all_pixels[clustering_results_filter]

    # Aligning the cluster numbers to 1..n having the value 0 for "not to use" ones
    selected_pixels[:, 2] = selected_pixels[:, 2] + 1

    #
    # Cluster erosion
    #
    # Simulate a full image having value 1 for the position belonging to the cluster
    x_size = application_context.s2_images.x_size
    y_size = application_context.s2_images.y_size
    result_matrix = numpy.zeros((x_size * y_size), numpy.uint8)
    result_matrix[selected_pixels[:, 1]] = 1
    # For erosion, it is required a 2D matrix
    result_matrix = result_matrix.reshape((x_size, y_size), order='F')

    # Perform erosion, depending on configuration
    if configuration.SAMPLES_CLUSTER_EROSION_TYPE == "BINARY_EROSION":
        application_context.logger.info("Performing clusters erosion using "
                                        + configuration.SAMPLES_CLUSTER_EROSION_TYPE)
        result_matrix = utils.erode_matrix_binary_same_values(
            result_matrix, disk_radius=configuration.SAMPLES_CLUSTER_EROSION_RADIUS)
    elif configuration.SAMPLES_CLUSTER_EROSION_TYPE == "SMALL_OBJECTS":
        application_context.logger.info("Performing clusters erosion using "
                                        + configuration.SAMPLES_CLUSTER_EROSION_TYPE)
        result_matrix = utils.erode_matrix_small_objects_same_values(
            result_matrix, minimum_size=configuration.SAMPLES_CLUSTER_EROSION_SMALL_OBJECTS_SIZE)
    else:
        application_context.logger.info("No cluster erosion configured")

    # Transforming the matrix for having the same indexes for altering the cluster positions
    result_matrix = result_matrix.reshape((x_size * y_size), order='F')
    # Updating the cluster positions for the remaining positions after the erosion (checking which indexes are still in
    # the post-erosion matrix)
    selected_pixels = selected_pixels[numpy.isin(selected_pixels[:, 1], numpy.where(result_matrix > 0))]

    return selected_pixels


def main(application_context):
    """
        Execute the 'STEP3 Extract sample from map'
        :param application_context: The application context
        :type application_context: ApplicationContext
        :return: A dictionary representing, for each SEOM class code, the sample pixels
        :rtype: dict
    """
    application_context.logger.info("Starting execution of 'STEP3 Extract sample from map'")
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context)

    # Getting the sample image
    samples_matrix = get_samples_matrix(application_context)

    # Reading the cloud cover image
    cloud_free_converted_clc_map = get_cloud_free_clc_matrix(application_context)

    # Computing the indexes matrix (NDVI, NDWI...)
    application_context.logger.info("Computing the indexes matrix (NDVI, NDWI...) for the samples matrix")
    indexes_matrix = get_indexes_matrix(samples_matrix, application_context.logger)
    samples_matrix = None # Freeing memory

    # Normalize the data (i.e. scaling among [0,1])
    application_context.logger.info("Normalizing the indexes matrix")
    # Compute the minimum along each column
    min_values = numpy.amin(indexes_matrix, axis=0)
    # Compute the maximum along each column
    max_values = numpy.amax(indexes_matrix, axis=0)
    indexes_matrix_normalized = utils.get_normalized_image_matrix(indexes_matrix, min_values, max_values)
    indexes_matrix = None  # Freeing memory
    min_values = None
    max_values = None

    # The result
    selected_pixels_for_class = dict()

    # Iterate over all the classes
    for seom_class in configuration.seomClcLegend.seom_classes:
        # Skipping the 'Unknown' class
        if seom_class.class_value == 0:
            continue

        application_context.logger.info("Generating samples for class {} ({})"
                                        .format(str(seom_class.class_value), seom_class.class_name))

        # Retrieving the indexes in the clc matrix having the current class value
        clc_indexes_for_class = numpy.where(cloud_free_converted_clc_map[:, 0] == seom_class.class_value)[0]

        # Performing the selection among the available values
        selected_pixels = generate_samples(
            application_context,
            indexes_matrix_normalized,
            clc_indexes_for_class,
            seom_class.class_value,
            seom_class.number_samples_clusters,
            seom_class.number_samples_clusters_to_keep
        )

        # Storing the data
        selected_pixels_for_class[seom_class.class_value] = selected_pixels

        if application_context.input_parameters.is_to_save_intermediate_outputs:
            destination_file_path = os.path.join(
                application_context.input_parameters.output_path,
                configuration.SAMPLES_OUTPUT_FILE_NAME.format(seom_class.class_value))
            application_context.logger.info("Saving step result for class {} ({}) into {}".format(
                seom_class.class_value,
                seom_class.class_name,
                destination_file_path))
            numpy.savetxt(destination_file_path, selected_pixels, delimiter=",", fmt="%d")

            # Additional elements for debugging purposes
            if configuration.SAVE_CLUSTER_IMAGES and len(selected_pixels) > 0:
                x_size = application_context.s2_images.x_size
                y_size = application_context.s2_images.y_size
                result_matrix = numpy.zeros((x_size, y_size), numpy.uint8)
                result_matrix = result_matrix.reshape(x_size * y_size, order='F')
                for i in range(1, seom_class.number_samples_clusters + 1):
                    result_matrix[selected_pixels[selected_pixels[:, 2] == i][:, 1]] = i
                result_matrix = result_matrix.reshape(x_size, y_size, order='F')
                destination_file_path = os.path.join(
                    application_context.input_parameters.output_path,
                    'clusters_' + str(seom_class.class_value) + '.tif'
                )
                utils.save_matrix_as_geotiff(
                    result_matrix,
                    destination_file_path,
                    None,
                    configuration.IMAGES_OUTPUT_GDAL_TYPE,
                    application_context.s2_images.geo_transform,
                    application_context.s2_images.projection
                )

    application_context.logger.info("Execution of 'STEP3 Extract sample from map' successfully completed")
    return selected_pixels_for_class


if __name__ == "__main__":
    # Remember to check the THREADs used, because each one will use its own memory.
    # Using 4 threads on T32TPS took like 22 GB of RAM computing Conifers.
    application_context = ApplicationContext()
    output_path = "/mnt/hgfs/seomData/s2/2018_T33STC/output"
    log_configuration = logging_utils.get_log_configuration(
        os.path.join("../../", configuration.LOGGING_CLASSIFICATION_CONFIGURATION_FILE_PATH))
    logging_utils.override_log_file_handler_path(log_configuration, 'file', output_path)
    logging_utils.initialize_log(log_configuration)
    application_context.logger = logging_utils.get_logger(configuration.CLASSIFICATION_LOGGER_NAME)
    application_context.input_parameters.is_to_save_intermediate_outputs = True
    application_context.input_parameters.output_path = output_path
    application_context.input_parameters.s2_data_path = "/mnt/hgfs/seomData/s2/2018_T33STC/s2_images"
    application_context.input_parameters.s2_samples_image_name = "MSIL2A_20180712T095031_N0206_R079_T33STC.tif"
    application_context.input_parameters.cloud_mask_image_path = "/mnt/hgfs/seomData/s2/2018_T33STC/c_masks/Cmask_20180712.tif"
    application_context.s2_images = Sentinel2Images()
    application_context.s2_images.path = application_context.input_parameters.s2_data_path
    application_context.s2_images.name_matrix_map = dict.fromkeys([
        application_context.input_parameters.s2_samples_image_name
    ])
    images = utils.retrieve_images_gdal([os.path.join(output_path, "clc_converted.tif")])
    application_context.clc_map_converted_matrix = utils.transform_image_to_matrix(
        images[0],
        target_type=configuration.IMAGES_CLC_NUMPY_TYPE)

    # Acquisition of the reference image parameters
    try:
        path = os.path.join(application_context.s2_images.path,
                            application_context.input_parameters.s2_samples_image_name)
        application_context.logger.info("Reading the sample image " + path)

        candidates = utils.retrieve_images_gdal([path])
        reference_image = candidates[0]
        application_context.s2_images.geo_transform = reference_image.GetGeoTransform()
        application_context.s2_images.projection = reference_image.GetProjection()
        application_context.s2_images.x_size = reference_image.RasterXSize
        application_context.s2_images.y_size = reference_image.RasterYSize
        application_context.s2_images.bands_size = reference_image.RasterCount
    except SeomException as e:
        raise
    except Exception as e:
        raise SeomException("Unable to retrieve the Sentinel2 samples image", e)

    try:
        main(application_context)
    except Exception as e:
        if application_context is not None and application_context.logger is not None:
            application_context.logger.critical(traceback.format_exc())
        else:
            print(str(traceback.format_exc()))
