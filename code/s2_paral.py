# Step2 Convert the Corine Land Cover map to 16 classes accordingly with the Land Cover Classification System
#!/usr/bin/env python
import os
import traceback
import numpy
import argparse
from configurations import configuration
from modules import utils, logging_utils
from modules.exceptions.SeomException import SeomException
from modules.models.application_context import ApplicationContext
from modules.models.input_parameters import InputParameters
from modules.models.seom_clc_map_legend import SeomClcMapLegend

def validate_step_data(application_context, seom_clc_legend):
    """
        Validates all the data required by the step. Not all the given content will be validated, just the ones for
        the current step
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param seom_clc_legend: The SEOM CLC (Corine Land Cover) map legend
        :type seom_clc_legend: SeomClcMapLegend
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(application_context.input_parameters, InputParameters), \
        "Wrong input parameter type for 'input_parameters', expected InputParameters"

    # Either is set the application_context.clc_original_image or the path has to exists
    if application_context.clc_original_image is not None:
        assert isinstance(application_context.clc_original_image, numpy.ndarray), \
            "The 'application_context' 'clc_original_image' has wrong type, expected numpy.ndarray"
    elif not os.path.exists(application_context.input_parameters.corine_land_cover_data_path) or \
            not os.path.isfile(application_context.input_parameters.corine_land_cover_data_path):
        raise SeomException("The given path for Land Cover data doesn't exist or it is wrong")

    assert isinstance(seom_clc_legend, SeomClcMapLegend), \
        "Wrong input parameter type for 'seom_clc_legend', expected SeomClcMapLegend"
    
def clean_clc_map(clc_map_matrix):
    """
        Clean the Corine Land Cover map from values not coherent with the Land Cover Classification System (LCCS)
        :param clc_map_matrix: The Corine Land Cover map matrix
        :type clc_map_matrix: numpy.ndarray
        :return: The cleaned Corine Land Cover map matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(clc_map_matrix, numpy.ndarray), \
        "Wrong input parameter type for 'clc_map_matrix', expected numpy.ndarray"

    # Filter the CLC classes
    cleaning_filter = None

    for clc_class in configuration.clc_map_legend:
        if cleaning_filter is None:
            cleaning_filter = clc_map_matrix != clc_class.class_value
        else:
            cleaning_filter = numpy.logical_and(cleaning_filter, clc_map_matrix != clc_class.class_value)

    # Set to 0 all values different from the configured ones
    clc_map_matrix[cleaning_filter] = 0

    return clc_map_matrix

def convert_clc_map(clc_map_matrix, seom_clc_legend):
    """
        Convert the Corine Land Cover map to 16 classes accordingly with the Land Cover Classification System (LCCS)
        :param clc_map_matrix: The Corine Land Cover map matrix
        :type clc_map_matrix: numpy.ndarray
        :param seom_clc_legend: The SEOM CLC (Corine Land Cover) map legend. This represents the map among the official
        CLC legend and the one which will be used by the SEOM project
        :type seom_clc_legend: SeomClcMapLegend
        :return: The converted Corine Land Cover map matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(clc_map_matrix, numpy.ndarray), \
        "Wrong input parameter type for 'clc_map_matrix', expected numpy.ndarray"

    # Filter for all remaining classes
    all_others_filter = None

    # Process all the elements in the SEOM legend
    for seom_clc_class in seom_clc_legend.seom_classes:
        original_classes_filter = None

        # In case there is just one original CLC class, simple use it. Otherwise compose considering all of them
        if len(seom_clc_class.original_classes) == 1:
            original_classes_filter = clc_map_matrix == seom_clc_class.original_classes[0].class_value
        else:
            # Processing the multiple original classes
            for clc_original_class in seom_clc_class.original_classes:
                if original_classes_filter is None:
                    original_classes_filter = clc_map_matrix == clc_original_class.class_value
                    continue

                original_classes_filter = numpy.logical_or(original_classes_filter, clc_map_matrix == clc_original_class.class_value)

        # Filtering the data with the build conditions and set those values to the target SEOM class value
        clc_map_matrix[original_classes_filter] = seom_clc_class.class_value

        # Updating the filter for the remaining classes
        if all_others_filter is None:
            all_others_filter = clc_map_matrix != seom_clc_class.class_value
        else:
            all_others_filter = numpy.logical_and(all_others_filter, clc_map_matrix != seom_clc_class.class_value)

    # Set to 0 all values different from the configured ones
    clc_map_matrix[all_others_filter] = 0

    return clc_map_matrix

def main(application_context):
    """
        Execute the 'STEP2 Preprocessing Corine Land Cover data'
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    application_context.logger.info("Starting execution of 'STEP2 Preprocessing Corine Land Cover data'")
    assert isinstance(application_context, ApplicationContext)\
        , "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context, configuration.seomClcLegend)

    clc_map_matrix = None
    if application_context.clc_original_image is not None:
        application_context.logger.info("The original CLC is already in memory")
        clc_map_matrix = application_context.clc_original_image
    else:
        application_context.logger.info("Acquiring the input data from " +
                                        application_context.input_parameters.corine_land_cover_data_path)
        results = utils.retrieve_images_gdal([application_context.input_parameters.corine_land_cover_data_path])
        if results is None or len(results) == 0:
            raise SeomException("Unable to retrieve the Corine Land Cover map")
        clc_image = results[0]
        clc_map_matrix = utils.transform_image_to_matrix(clc_image, configuration.IMAGES_CLC_NUMPY_TYPE)

    application_context.logger.info("Executing erosion")
    eroded_clc_map_matrix = utils.erode_matrix_binary_same_values(clc_map_matrix,
                                                                  disk_radius=configuration.IMAGES_CLC_EROSION_RADIUS)

    application_context.logger.info("Cleaning the Corine Land Cover from not-existing CLC classes "
                                    "(based on configuration)")
    cleaned_clc_map_matrix = clean_clc_map(eroded_clc_map_matrix)
    eroded_clc_map_matrix = None

    seom_class_codes_size = len(configuration.seomClcLegend.get_seom_class_codes())
    application_context.logger.info("Converting the Corine Land Cover into {}+1 classes"
                                    .format(str(seom_class_codes_size)))
    clc_map_converted_matrix = convert_clc_map(cleaned_clc_map_matrix, configuration.seomClcLegend)

    if application_context.input_parameters.is_to_save_intermediate_outputs:
        destination_file_path = os.path.join(
            application_context.input_parameters.output_path,
            configuration.IMAGES_CLC_PREPROCESSING_OUTPUT_FILE_NAME)
        application_context.logger.info("Saving step result into " + destination_file_path)
        utils.save_matrix_as_geotiff(
            clc_map_converted_matrix,
            destination_file_path,
            clc_image, # TODO In case the image is in memory, need to retrieve the data for projecting-transforming
            configuration.IMAGES_OUTPUT_GDAL_TYPE,
            colors_maps_array=[configuration.seomClcLegend.get_colors_map()]
        )

    application_context.logger.info("Execution of 'STEP2 Preprocessing Corine Land Cover data' successfully completed")
    return clc_map_converted_matrix

if __name__ == "__main__":
    application_context = ApplicationContext()
    
    #set the task id
    parser = argparse.ArgumentParser(description='Setting argument parser')
    parser.add_argument("--id_acquisition", help="Select id of the acquisition for s2", type=int, default=1)
    args = parser.parse_args()
    application_context.id_acquisition = args.id_acquisition
    
    '''
    # plan to read from a list later (where to set the thread )
    if application_context.id_acquisition == 1:
        application_context.input_parameters.corine_land_cover_data_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/corinemap/31UGS/31UGS.tif"
        output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output/31UGS"
    if application_context.id_acquisition == 2:
        application_context.input_parameters.corine_land_cover_data_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/corinemap/31UFT/31UFT.tif"
        output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output/31UFT"
    '''
    #tile_list = ['31UES', '31UET', '31UFS', '31UFT', '31UFU', '31UFV', '31UGS', '31UGU', '31UGV', '31UGT']
    tile_list = ['31UFS', '31UFT']
    year = "2018"
    for i in range(len(tile_list)):
        if application_context.id_acquisition == i:            
            tile = tile_list[i]
            input_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/corinemap/netherland/" + tile + ".tif"
            output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/new/output/test" + tile + "/" + year
            
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_configuration = logging_utils.get_log_configuration(
        os.path.join("../../", configuration.LOGGING_CLASSIFICATION_CONFIGURATION_FILE_PATH))
    logging_utils.override_log_file_handler_path(log_configuration, 'file', output_path)
    logging_utils.initialize_log(log_configuration)
    application_context.logger = logging_utils.get_logger(configuration.CLASSIFICATION_LOGGER_NAME)
    utils.initialize_legend_colors(configuration.seomClcLegend, os.path.join("../../", configuration.SEOM_COLOR_MAP_PATH))
    application_context.input_parameters.is_to_save_intermediate_outputs = True
    application_context.input_parameters.output_path = output_path
    application_context.input_parameters.is_to_save_intermediate_outputs = True
    application_context.input_parameters.corine_land_cover_data_path = input_path
    try:
        main(application_context)
    except Exception as e:
        if application_context is not None and application_context.logger is not None:
            application_context.logger.critical(traceback.format_exc())
        else:
            print(str(traceback.format_exc()))
