#!/usr/bin/env python
import os
import traceback

from configurations import configuration
from modules import utils, logging_utils
from modules.exceptions.SeomException import SeomException
from modules.models.application_context import ApplicationContext
from modules.models.input_parameters import InputParameters
from modules.models.sentinel2_images import Sentinel2Images


def validate_step_data(application_context):
    """
        Validates all the data required by the step. Not all the given content will be validated, just the ones for
        the current step
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(application_context.input_parameters, InputParameters), \
        "Wrong input parameter type for 'input_parameters', expected InputParameters"

    # Either is set the application_context.s2_images or the path has to exists
    if application_context.s2_images is not None:
        assert isinstance(application_context.s2_images, Sentinel2Images), \
            "The 'application_context' 's2_images' has wrong type, expected Sentinel2Images"
    elif not os.path.exists(application_context.input_parameters.s2_data_path) \
            or not os.path.isdir(application_context.input_parameters.s2_data_path):
        raise SeomException("The given path for Sentinel2 data doesn't exist or it is wrong")

    # TODO Check the maximum size for the value in the image like image.GetRasterBand(1).GetStatistics(True, True)


def main(application_context):
    """
        Execute the 'STEP1 Preprocessing Sentinel2 data'
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    application_context.logger.info("Starting execution of 'STEP1 Preprocessing Sentinel2 data'")
    assert isinstance(application_context, ApplicationContext)\
        , "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context)

    if application_context.s2_images is not None and \
            application_context.s2_images.name_matrix_map is not None and \
            all(image is not None for image in application_context.s2_images.name_matrix_map.values()):
        application_context.logger.info("Sentinel2 data images already in memory")
        sentinel2_images = application_context.s2_images
    else:
        application_context.logger.info("Acquiring the input data from " + application_context.input_parameters.s2_data_path)
        s2_images_paths = utils.get_images_paths_from_folder(application_context.input_parameters.s2_data_path)
        if s2_images_paths is None or len(s2_images_paths) == 0:
            raise SeomException("No Sentinel2 images found in the given path")

        # Reading the first for acquiring metadata
        s2_images = utils.retrieve_images_gdal([s2_images_paths[0]])
        if s2_images is None or len(s2_images) == 0:
            raise SeomException("Cannot retrieve Sentinel2 images")

        # Defining the structure
        sentinel2_images = Sentinel2Images()
        sentinel2_images.path = application_context.input_parameters.s2_data_path
        sentinel2_images.projection = s2_images[0].GetProjection()
        sentinel2_images.geo_transform = s2_images[0].GetGeoTransform()
        sentinel2_images.x_size = s2_images[0].RasterXSize
        sentinel2_images.y_size = s2_images[0].RasterYSize
        sentinel2_images.bands_size = s2_images[0].RasterCount
        # Initializing the map of images
        s2_images_file_names = [os.path.basename(element) for element in s2_images_paths]
        sentinel2_images.name_matrix_map = dict.fromkeys(s2_images_file_names, None)

    application_context.logger.debug("The Sentinel2 images available are: " +
                                     str([name for name in sentinel2_images.name_matrix_map]))

    return sentinel2_images


if __name__ == "__main__":
    application_context = ApplicationContext()
    output_path = "/mnt/hgfs/project/data/2018_T32TPS/output/seomLegendChange"
    log_configuration = logging_utils.get_log_configuration(
        os.path.join("../../", configuration.LOGGING_CLASSIFICATION_CONFIGURATION_FILE_PATH))
    logging_utils.override_log_file_handler_path(log_configuration, 'file', output_path)
    logging_utils.initialize_log(log_configuration)
    application_context.logger = logging_utils.get_logger(configuration.CLASSIFICATION_LOGGER_NAME)
    application_context.input_parameters.is_to_save_intermediate_outputs = True
    application_context.input_parameters.output_path = output_path
    application_context.input_parameters.s2_data_path = "/mnt/hgfs/project/data/2018_T32TPS/s2_images"

    try:
        main(application_context)
    except Exception as e:
        if application_context is not None and application_context.logger is not None:
            application_context.logger.critical(traceback.format_exc())
        else:
            print(str(traceback.format_exc()))
