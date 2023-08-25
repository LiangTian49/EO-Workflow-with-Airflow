import datetime
import os

import gdal
import numpy

from configurations import configuration
from modules import utils
from modules.models.application_context import ApplicationContext
from modules.models.input_parameters import InputParameters
from modules.models.sentinel2_images import Sentinel2Images


# TODO Refactor for being as generic as check_or_load_predicted_images
def check_or_load_cloud_masks(application_context, load=True):
    """
        Check the cloud masks to be in the application context or load them.
        Expected to load 0.242 GB * 6 ~ 1.45 GB.
        It also checks (and set if missing) the cloud mask path
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param load: True for loading the required images
        :type load: bool
        :return: True in case all the images are already in the application_context or have been loaded
        :rtype: bool
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(application_context.input_parameters, InputParameters), \
        "Wrong input parameter type for 'application_context' 'input_parameters', expected 'InputParameters'"
    assert isinstance(application_context.input_parameters.s2_samples_image_name, str), \
        "Wrong input parameter type for 'application_context' 'input_parameters' 's2_samples_image_name', expected str"
    assert isinstance(application_context.input_parameters.cloud_mask_image_path, str), \
        "Wrong input parameter type for 'application_context' 'input_parameters' 'cloud_mask_image_path', expected str"
    assert isinstance(application_context.s2_images, Sentinel2Images), \
        "Wrong input parameter type for 'application_context' 's2_images', expected 'Sentinel2Images'"

    # Getting the current tile name and the date
    tile_name, tile_year, tile_month, tile_day = \
        utils.get_tile_date_by_s2_file_name(application_context.input_parameters.s2_samples_image_name)
    tile_date = datetime.date(tile_year, tile_month, tile_day).strftime('%Y%m%d')
    tile_folder_name = str(tile_year) + "_" + tile_name

    # Check in case there isn't the image
    if application_context.s2_images.name_cloud_covers_matrix_map is None or \
            len(application_context.s2_images.name_cloud_covers_matrix_map) == 0:
        # In case the load is not require, just return
        if not load:
            return False
    else:
        # Check the given in-memory images to be compliant the expectation
        map = application_context.s2_images.name_cloud_covers_matrix_map
        assert all(isinstance(map[element], numpy.ndarray)
                   for element in map), \
            "Wrong input parameter type for 'name_cloud_covers_matrix_map', expected map of numpy.ndarray"
        return True

    # Preparing data structure
    application_context.s2_images.name_cloud_covers_matrix_map = dict()

    # Retrieving data
    s2_name_cloud_mask_path_map = utils.get_s2_cloud_masks_images_file_paths_map(application_context)
    for s2_image_name in s2_name_cloud_mask_path_map:
        cloud_mask_path = s2_name_cloud_mask_path_map[s2_image_name]
        application_context.logger.debug("Loading the image " + cloud_mask_path)
        cloud_mask_image = utils.retrieve_image_matrix(cloud_mask_path, target_type=configuration.IMAGES_S2_NUMPY_TYPE)
        application_context.s2_images.name_cloud_covers_matrix_map[s2_image_name] = cloud_mask_image

    return True


# TODO Refactor for being as generic as check_or_load_predicted_images
def check_or_load_final_predicted_image(application_context, load=True):
    """
        Check the final predicted image to be in the application context or load it.
        Expected to load 0.242 GB.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param load: True for loading the required image
        :type load: bool
        :return: True in case all the image is already in the application_context or has been loaded
        :rtype: bool
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    # Check in case there isn't the image
    if application_context.predicted_image is None:
        # In case the load is not require, just return
        if not load:
            return False
    else:
        # Check the given in-memory images to be compliant the expectation
        assert isinstance(application_context.predicted_image, numpy.ndarray), \
            "Wrong input parameter type for 'predicted_image', expected numpy.ndarray"
        return True

    # Getting the current tile name and its year
    tile_data = utils.get_tile_date_by_s2_file_name(application_context.input_parameters.s2_samples_image_name)
    tile_name = tile_data[0]
    tile_year = tile_data[1]
    prediction_image_file_name = str(tile_year) + "_" + tile_name
    file_path = os.path.join(
        application_context.input_parameters.output_path,
        configuration.CLASSIFICATION_OUTPUT_FINAL_FILE_NAME.format(prediction_image_file_name))

    if not os.path.isfile(file_path):
        return False

    # Load the image
    application_context.predicted_image = utils.retrieve_image_matrix(file_path,
                                                                      target_type=configuration.IMAGES_S2_NUMPY_TYPE)

    return True


def check_or_load_predicted_images(application_context, load=True, target_epsg_code=None):
    """
        Check the predicted images to be in the application context or load them.
        Expected to load 0.242 GB * configuration.SAMPLES_STRATIFIED_TRIALS size (e.g. for 5 images: ~1.2 GB)
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param load: True for loading the required images
        :type load: bool
        :return: True in case all the images are already in the application_context or have been loaded
        :rtype: bool
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    logger = application_context.logger

    # Check in case there aren't any images
    if application_context.predicted_images is None or len(application_context.predicted_images) == 0:
        # In case the load is not require, just return
        if not load:
            return False
    else:
        # Check the given in-memory images to be compliant the expectation
        assert isinstance(application_context.predicted_images, list), \
            "Wrong input parameter type for 'predicted_images', expected list"
        assert all(isinstance(element, numpy.ndarray) for element in application_context.predicted_images), \
            "Wrong input parameter type for 'predicted_images', expected list of numpy.ndarray"
        assert len(application_context.predicted_images) == configuration.SAMPLES_STRATIFIED_TRIALS, \
            "The number of predicted images doesn't match the configured SAMPLES_STRATIFIED_TRIALS number"
        return True

    # Images path
    images_path = None

    # Checking data structure for holding the geo-transformation and projection for the tile images
    if application_context.s2_images is None:
        application_context.s2_images = Sentinel2Images()

    if logger is not None:
        logger.debug("Retrieving the predicted images from local path")

    # Checking the proper local path
    images_path = application_context.input_parameters.output_path
    if os.path.isdir(os.path.join(images_path, configuration.DATA_OUTPUT_FOLDER_NAME)):
        images_path = os.path.join(images_path, configuration.DATA_OUTPUT_FOLDER_NAME)

    has_final_classified = False
    has_final_postprocessing = False
    has_final_postproc_overlaps = False
    for file_name in os.listdir(images_path):
        if not has_final_classified:
            has_final_classified = utils.is_valid_classified_image_name(file_name, 'final-classified')

        if not has_final_postprocessing:
            has_final_postprocessing = utils.is_valid_classified_image_name(file_name, 'final-postprocessing')

        if not has_final_postproc_overlaps:
            has_final_postproc_overlaps = utils.is_valid_classified_image_name(file_name, 'final-postprocessing-overlaps')

    # Actual loading the images
    application_context.predicted_images = []
    for file_name in os.listdir(images_path):
        file_path = os.path.join(
            application_context.input_parameters.output_path, file_name)

        is_final_classified = utils.is_valid_classified_image_name(file_name, 'final-classified')
        is_final_postprocessing = utils.is_valid_classified_image_name(file_name, 'final-postprocessing')
        is_final_postproc_overlaps = utils.is_valid_classified_image_name(file_name, 'final-postprocessing-overlaps')

        # Skipping those files not following the expected name pattern
        if not os.path.isfile(file_path) \
                or (not utils.is_valid_classified_image_name(file_name, 'trial') \
                    and not is_final_classified
                    and not is_final_postprocessing
                    and not is_final_postproc_overlaps):
            continue

        # Updating the application context
        if logger is not None:
            logger.debug("Loading " + file_path)

        to_delete = False
        if target_epsg_code is not None:
            gdal_images = utils.retrieve_images_gdal([file_path])
            img_epsg_code = utils.get_epsg_code_from_gdal_projection(gdal_images[0].GetProjection())
            if img_epsg_code != str(target_epsg_code):
                if logger is not None:
                    logger.debug("Reprojecting to " + str(target_epsg_code))

                old_path = file_path
                file_path = os.path.join(os.path.dirname(file_path), "tmp.tif")
                gdal.Warp(file_path, old_path, dstSRS='EPSG:' + str(target_epsg_code))
                to_delete = True

        predicted_image, geo_transform, projection = \
            utils.retrieve_image_matrix_with_spatial_references(file_path,
                                                                target_type=configuration.IMAGES_PREDICTED_NUMPY_TYPE)

        # Remove the file which is the reprojected
        if to_delete:
            os.remove(file_path)

        # In case it is a final prediction, assign value just in case no previous value is set - so to keep the
        # postprocessing result if any
        if (is_final_classified and (not has_final_postprocessing and not has_final_postproc_overlaps)) or \
            (is_final_postprocessing and not has_final_postproc_overlaps) or \
            is_final_postproc_overlaps:
            application_context.predicted_image = predicted_image

        if not is_final_classified and not is_final_postprocessing and not is_final_postproc_overlaps:
            application_context.predicted_images.append(predicted_image)

        if application_context.s2_images.geo_transform is None:
            application_context.s2_images.geo_transform = geo_transform

        if application_context.s2_images.projection is None:
            application_context.s2_images.projection = projection

    if len(application_context.predicted_images) == 0 and logger is not None:
        logger.debug("There aren't any predicted images in the output path " + str(images_path))

    # Loading
    if len(application_context.predicted_images) == 0 and configuration.DESTINATION_REMOTE:
        images_path = application_context.input_parameters.temporary_path
        remote_server_password = application_context.remote_server_password
        if logger is not None:
            logger.debug("Retrieving the predicted images from remote source to " + str(images_path))

        # Retrieve the images from remote to temporary folder
        predicted_images, predicted_image, geo_transform, projection = \
            utils.retrieve_classified_images_from_remote(application_context, images_path, remote_server_password,
                                                         logger)

        application_context.predicted_images = predicted_images
        application_context.predicted_image = predicted_image

        if application_context.s2_images.geo_transform is None:
            application_context.s2_images.geo_transform = geo_transform

        if application_context.s2_images.projection is None:
            application_context.s2_images.projection = projection

        # TODO? Delete the temporary images
        if logger is not None:
            logger.debug("Deleting temporary data")
        pass

    return len(application_context.predicted_images) > 0


# TODO Refactor for being as generic as check_or_load_predicted_images (and transform it to a check_or_load)
def get_s2_image_matrix(application_context, s2_image_name, bands=None):
    """
        Retrieve the given image either from application context or properly loads it.
        Expected size of 2.4 GB
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param s2_image_name: The Sentinel 2 image name to retrieve
        :type s2_image_name: str
        :param bands: The bands to retrieve, if 'None' all the bands will be retrieved. It is 0 based, so the first band
        has the identifier 0.
        :type bands: list of int
        :return: The requested s2 image
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(application_context.s2_images, Sentinel2Images), \
        "Wrong input parameter type for 'application_context' 's2_images', expected Sentinel2Images"
    assert isinstance(application_context.s2_images.name_matrix_map, dict), \
        "Wrong input parameter type for 'application_context' 's2_images' 'name_matrix_map', expected dict"

    if s2_image_name in application_context.s2_images.name_matrix_map and \
            application_context.s2_images.name_matrix_map[s2_image_name] is not None:
        if bands is not None:
            return application_context.s2_images.name_matrix_map[s2_image_name][:, :, bands]

        return application_context.s2_images.name_matrix_map[s2_image_name]

    image_path = os.path.join(
        application_context.input_parameters.s2_data_path,
        s2_image_name
    )
    image_matrix = utils.retrieve_image_matrix(image_path, configuration.IMAGES_S2_NUMPY_TYPE, bands)
    return image_matrix