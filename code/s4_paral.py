#!/usr/bin/env python
import os
import traceback
import numpy
import argparse
from sklearn.preprocessing import MinMaxScaler

from configurations import configuration
from modules import utils, logging_utils
from modules.exceptions import SeomException
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
        "The input parameter for the Sentinel2 image for computing the samples is expected to be a string " \
        "representing a file name"
    
def get_map_usable_classes(clc_converted_map, logger=None):
    """
        Retrieve the map with only the usable classes, which are those greater than 0 (the 'unknown' class in the CLC)
        and lesser than the maximum value of the configured class codes in the SEOM legend.
        :param clc_converted_map: The Corine Land Cover (CLC) map converted for representing the SEOM classes
        :type clc_converted_map: numpy.ndarray
        :param logger: The logger
        :type logger: logging.Logger
        :return: The CLC map converted with only usable classes
        :rtype: numpy.ndarray
    """
    assert isinstance(clc_converted_map, numpy.ndarray),\
        "Wrong type for 'clc_converted_map', expected numpy.ndarray"

    x_size = clc_converted_map.shape[0]
    y_size = clc_converted_map.shape[1]
    # Using order 'F' (Fortran ~ column major) for keeping consistency with MATLAB. Converting from matrix to array
    clc_map_array = numpy.reshape(clc_converted_map, x_size * y_size, order='F')
    # Getting the maximum class code in the map
    clc_map_maximum = numpy.amax(clc_map_array)
    # Getting the maximum class code from configuration
    legend_maximum = max([c.class_value for c in configuration.seomClcLegend.seom_classes])
    if clc_map_maximum > legend_maximum:
        logger.warning("The CLC converted map has a maximum greater than the legend one: " + str(clc_map_maximum))

    # Getting only the usable elements (belonging to the SEOM class which will be used)
    # Removing the '0' because it will compromise the prior calculation, other than it is unusable
    usable_class_values_condition = numpy.logical_and(clc_converted_map > 0, clc_converted_map <= legend_maximum)
    clc_map_usable_classes = clc_converted_map[usable_class_values_condition]

    return clc_map_usable_classes, legend_maximum

def get_prior(clc_converted_map):
    """
        :param clc_converted_map: The Corine Land Cover (CLC) map converted for representing the SEOM classes
        :type clc_converted_map: numpy.ndarray
        :return: The prior for each class
        :rtype: numpy.ndarray
    """
    assert isinstance(clc_converted_map, numpy.ndarray),\
        "Wrong type for 'clc_converted_map', expected numpy.ndarray"

    # Retrieve the usable classes
    clc_map_usable_classes, clc_map_maximum = get_map_usable_classes(clc_converted_map)

    # Computing the occurrences by position (expected values to be from 1..n where n is the number of SEOM classes)
    seom_class_codes, seom_class_code_occurrences = numpy.unique(clc_map_usable_classes, return_counts=True)
    seom_class_codes_total_occurrences = numpy.sum(seom_class_code_occurrences)
    # There could be some classes not represented in the map. Adding them
    # NOTE: The i represent the position, while the actual class code would be i + 1. So in position 0 there is the
    # class code 1
    for i in range(0, int(clc_map_maximum)):
        class_code = i + 1
        if class_code in seom_class_codes:
            continue

        seom_class_codes = numpy.insert(seom_class_codes, i, class_code)
        seom_class_code_occurrences = numpy.insert(seom_class_code_occurrences, i, 0)

    # Compute the frequencies
    seom_class_code_frequencies = (seom_class_code_occurrences / seom_class_codes_total_occurrences) * 100

    # Computing the prior
    prior = numpy.zeros(len(seom_class_codes), dtype=configuration.IMAGES_CLC_NUMPY_TYPE)
    target_quantile = configuration.SAMPLES_PRIOR_QUANTILE
    min_occurrences = configuration.SAMPLES_PRIOR_MIN_OCCURRENCES_FOR_CLASS
    selection = seom_class_code_frequencies[seom_class_code_occurrences > min_occurrences]
    # Computing the mask for the elements over the quantile
    mask_high = seom_class_code_frequencies > numpy.quantile(selection, target_quantile)
    # Computing the mask for the elements equal or below the quantile
    mask_low = seom_class_code_frequencies <= numpy.quantile(selection, target_quantile)

    # Getting the constants
    multiplier = configuration.SAMPLES_PRIOR_MULTIPLIER
    higher_mod = configuration.SAMPLES_PRIOR_HIGH_MODIFIER
    lower_mod = configuration.SAMPLES_PRIOR_LOW_MODIFIER
    # Normalizing and rescaling
    min_max_scaler = MinMaxScaler()
    # The MinMaxScaler requires a matrix, so generate an array of array of single elements (e.g. [[1.2] [3.32]...])
    values = seom_class_code_frequencies[mask_high].reshape(-1, 1)
    # Computing the normalized prior for the mask_high reshaping the result to a single array and taking it
    prior[mask_high] = (min_max_scaler.fit_transform(values) * multiplier + higher_mod).reshape(1, values.size)[0]
    # The MinMaxScaler requires a matrix, so generate an array of array of single elements (e.g. [[1.2] [3.32]...])
    values = seom_class_code_frequencies[mask_low].reshape(-1, 1)
    # Computing the normalized prior for the mask_low reshaping the result to a single array and taking it
    prior[mask_low] = (min_max_scaler.fit_transform(values) * multiplier + lower_mod).reshape(1, values.size)[0]

    # Rounding
    prior = numpy.round(prior)
    # Resetting whenever there aren't enough occurrences in the class
    prior[seom_class_code_occurrences < 100] = 0

    return prior

def generate_trials(application_context, prior):
    """
        Compute the trials considering the 'stratified random sampling' approach
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param prior: The prior of the converted Corine Land Cover (CLC) map - the one having SEOM classes
        :type prior: numpy.ndarray
        :return: The trials results, the trials results classes
        :rtype: numpy.ndarray, numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext),\
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(prior, numpy.ndarray), "Wrong type for 'prior', expected numpy.ndarray"

    trials_result = []
    trials_result_classes = []

    for trial in range(0, configuration.SAMPLES_STRATIFIED_TRIALS): # configuration.SAMPLES_STRATIFIED_TRIALS was changes to 1
        # Instantiating the dictionary for the class samples result
        trials_result.append([])
        trials_result_classes.append([])

        for seom_class in configuration.seomClcLegend.seom_classes:
            # The unknown class is just to skip
            if seom_class.class_value == 0:
                continue

            # In case there aren't prior for the current class, skip it
            class_priors = prior[seom_class.class_value - 1]
            if class_priors == 0:
                continue

            # In case there aren't elements for the class
            if seom_class.class_value not in application_context.selected_pixels_for_class or\
               len(application_context.selected_pixels_for_class[seom_class.class_value]) == 0:
                continue

            candidate_pixels = application_context.selected_pixels_for_class[seom_class.class_value]
            sample_size = numpy.minimum(len(candidate_pixels), class_priors)

            # The candidate_pixels second column is the one with the image-array index to be used
            samples = numpy.random.choice(candidate_pixels[:, 1], int(sample_size))
            class_values = numpy.ones(int(sample_size),
                                      dtype=configuration.IMAGES_CLC_NUMPY_TYPE) * seom_class.class_value
            # Saving the status
            if trials_result[trial] is None or len(trials_result[trial]) == 0:
                trials_result[trial] = samples
                trials_result_classes[trial] = class_values
                continue

            trials_result[trial] = numpy.concatenate((trials_result[trial], samples), axis=0)
            trials_result_classes[trial] = numpy.concatenate((trials_result_classes[trial], class_values), axis=0)

    return trials_result, trials_result_classes

def generate_training_sets(application_context, trials_result, trials_result_classes):
    """
        Generate the training sets given the trials 'map' considering all the images in the time-series (expected to be
        'described' in the application_context.s2_images)
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param trials_result: The trials generated to use for generating the training sets
        :type trials_result: numpy.ndarray
        :param trials_result_classes: The seom class codes associated to the trials_result (i.e. the labels)
        :type trials_result_classes: numpy.ndarray
        :return: The training sets corresponding to the trials_result (with the first column to be the label of the data)
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext),\
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    training_sets = []

    s2_name_matrix_map = application_context.s2_images.name_matrix_map
    for trial in range(0, configuration.SAMPLES_STRATIFIED_TRIALS): # 
        application_context.logger.debug("Processing trial " + str(trial))

        # Initializing the current trial
        training_sets.append([])

        for s2_name in s2_name_matrix_map:
            if s2_name_matrix_map[s2_name] is None:
                path = os.path.join(application_context.s2_images.path, s2_name)
                s2_name_matrix_map[s2_name] = utils.retrieve_image_matrix(
                    path,
                    target_type=configuration.IMAGES_S2_NUMPY_TYPE)

            s2_image_matrix = s2_name_matrix_map[s2_name]
            x_size = s2_image_matrix.shape[0]
            y_size = s2_image_matrix.shape[1]
            bands = s2_image_matrix.shape[2]
            s2_image_array = numpy.reshape(s2_image_matrix, (x_size * y_size, bands), order='F')
            s2_image_matrix = None # Freeing memory
            trial_values = s2_image_array[trials_result[trial], :]
            s2_image_array = None  # Freeing memory

            if configuration.DEV_ENVIRONMENT:
                application_context.logger.debug("[DEV_ENVIRONMENT] Removing image " + s2_name)
                s2_name_matrix_map[s2_name] = None

            if training_sets[trial] is None or len(training_sets[trial]) == 0:
                # By using [:, None] the 1D array will be explicit as (n, 1) where n are the number of rows
                training_sets[trial] = numpy.concatenate((trials_result_classes[trial][:, None], trial_values), axis=1)
            else:
                training_sets[trial] = numpy.concatenate((training_sets[trial], trial_values), axis=1)

            trial_values = None  # Freeing memory

    return training_sets

def main(application_context):
    """
        Execute the 'STEP4 Stratified random sampling'
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    application_context.logger.info("Starting execution of 'STEP4 Stratified random sampling'")
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context)

    # Computing prior
    application_context.logger.info("Computing the prior on the CLC converted matrix")
    prior = get_prior(application_context.clc_map_converted_matrix)
    application_context.logger.debug("The priors are: " + str(prior))

    # Computing the trials
    application_context.logger.info("Computing the trials based on the SEOM classes considering the prior")
    trials_result, trials_result_classes = generate_trials(application_context, prior)

    # Actual samples acquisition
    application_context.logger.info("Computing the training sets based on the computed trials")
    training_sets = generate_training_sets(application_context, trials_result, trials_result_classes)

    if application_context.input_parameters.is_to_save_intermediate_outputs:
        for trial in range(0, configuration.SAMPLES_STRATIFIED_TRIALS): # configuration.SAMPLES_STRATIFIED_TRIALS was changes to 1
            destination_file_path = os.path.join(
                application_context.input_parameters.output_path,
                configuration.TRIALS_OUTPUT_FILE_NAME.format(trial + 1))
            application_context.logger.info("Saving training set {} in {}".format(
                trial,
                destination_file_path))
            numpy.savetxt(destination_file_path, training_sets[trial], delimiter=",", fmt="%d")

            if configuration.SAVE_TRIALS_IMAGES:
                destination_file_path = os.path.join(
                    application_context.input_parameters.output_path,
                    "trial_" + str(trial + 1) + ".tif"
                )
                x_size = application_context.s2_images.x_size
                y_size = application_context.s2_images.y_size
                result_matrix = numpy.zeros((x_size, y_size), numpy.uint8)
                result_matrix = result_matrix.reshape(x_size * y_size, order='F')
                result_matrix[trials_result[trial]] = trials_result_classes[trial]
                result_matrix = result_matrix.reshape(x_size, y_size, order='F')
                utils.save_matrix_as_geotiff(
                    result_matrix,
                    destination_file_path,
                    None,
                    configuration.IMAGES_OUTPUT_GDAL_TYPE,
                    application_context.s2_images.geo_transform,
                    application_context.s2_images.projection,
                    colors_maps_array=[configuration.seomClcLegend.get_colors_map()]
                )

    application_context.logger.info("Execution of 'STEP4 Stratified random sampling' successfully completed")
    return training_sets

if __name__ == "__main__":
    application_context = ApplicationContext()
    
    '''
    year = 2018
    tile_list = ['31UFT', '31UGS', '32ULC', '31UGU', '31UGV','31UFV', '31UFU','31UFS','31UES', '31UET']
    
    #set the task id
    parser = argparse.ArgumentParser(description='Setting argument parser')
    parser.add_argument("--id_acquisition", help="Select id of the acquisition", type=int, default=1)
    args = parser.parse_args()
    application_context.id_acquisition = args.id_acquisition
    
    application_context.input_parameters.s2_data_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/tif/" + tile_list[application_context.id_acquisition] + "/" + year
    s2_list = os.listdir(application_context.input_parameters.s2_data_path)
    s2_list.sort()
    application_context.input_parameters.s2_samples_image_name = s2_list[1]
    output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output
    '''
    #set the task id
    parser = argparse.ArgumentParser(description='Setting argument parser')
    parser.add_argument("--id_acquisition", help="Select id of the acquisition for s4", type=int, default=1)
    args = parser.parse_args()
    application_context.id_acquisition = args.id_acquisition
    
    '''
    # plan to read from a list later (where to set the thread )
    if application_context.id_acquisition == 1:
        application_context.input_parameters.s2_data_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/composite/31UGS/2018/"
        application_context.input_parameters.s2_samples_image_name = "2018_season1.tif"
        output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output/31UGS/"
        images = utils.retrieve_images_gdal([os.path.join(output_path, "clc_converted_31UGS.tif")])
    if application_context.id_acquisition == 2:
        application_context.input_parameters.s2_data_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/composite/31UFT/2018/"
        application_context.input_parameters.s2_samples_image_name = "2018_season1.tif"
        output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output/31UFT/"
        images = utils.retrieve_images_gdal([os.path.join(output_path, "clc_converted_31UFT.tif")])
    '''
    year = "2018"
    path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/input/"
    #tile_list = ['31UES', '31UET', '31UFS', '31UFT', '31UFU', '31UFV', '31UGS', '31UGU', '31UGV', '31UGT'] 
    tile_list = ['31UFS', '31UFT']
    for i in range(len(tile_list)):
        if application_context.id_acquisition == i:            
            tile = tile_list[i]
            #application_context.input_parameters.s2_data_path = path + tile + "/composite/" + year + "/"
            #output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output_12/" + tile + "/" + year
            application_context.input_parameters.s2_data_path = path + tile + "/tif/" + year + "/"
            s2_list = os.listdir(application_context.input_parameters.s2_data_path)
            application_context.input_parameters.s2_samples_image_name = s2_list[0]
            print('reference image is'+ application_context.input_parameters.s2_samples_image_name)
            
            #output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/new/output/" + tile + "/" + year
            output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/new/output/test" + tile + "/" + year
            images = utils.retrieve_images_gdal([os.path.join(output_path, "clc_converted.tif")])
            
    #application_context.input_parameters.s2_samples_image_name = "2018_season1.tif"
    
    log_configuration = logging_utils.get_log_configuration(
        os.path.join("../../", configuration.LOGGING_CLASSIFICATION_CONFIGURATION_FILE_PATH))
    logging_utils.override_log_file_handler_path(log_configuration, 'file', output_path)
    logging_utils.initialize_log(log_configuration)
    application_context.logger = logging_utils.get_logger(configuration.LOGGER_NAME)
    utils.initialize_legend_colors(configuration.seomClcLegend,
                                   os.path.join("../../", configuration.SEOM_COLOR_MAP_PATH))
    application_context.input_parameters.is_to_save_intermediate_outputs = True
    application_context.input_parameters.output_path = output_path
    application_context.s2_images = Sentinel2Images()
    application_context.s2_images.path = application_context.input_parameters.s2_data_path
    
    s2_list = os.listdir(application_context.input_parameters.s2_data_path)
    s2_list.sort()
    application_context.s2_images.name_matrix_map = dict.fromkeys(s2_list)

    #application_context.s2_images.name_matrix_map = dict.fromkeys(['2018_season1.tif', '2018_season2.tif', '2018_season3.tif'])
    application_context.clc_map_converted_matrix = utils.transform_image_to_matrix(images[0], target_type=numpy.int8)
    application_context.selected_pixels_for_class = dict()
    seom_classes_to_load = range(1, 12)
    for seom_class in configuration.seomClcLegend.seom_classes:
        # Skipping the 'Unknown' class
        if seom_class.class_value == 0:
            continue

        # For testing there may not be loading all the classes
        if seom_class.class_value not in seom_classes_to_load:
            continue

        class_value = seom_class.class_value
        file_path = os.path.join(
                application_context.input_parameters.output_path,
                configuration.SAMPLES_OUTPUT_FILE_NAME.format(seom_class.class_value))
        application_context.selected_pixels_for_class[class_value] = numpy.loadtxt(file_path, delimiter=",", dtype=int)

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

