import multiprocessing
import os
import time
import traceback

import numpy
from scipy.spatial.ckdtree import cKDTree

from configurations import configuration
from modules import logging_utils, utils, resources
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
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    assert application_context.s2_images is not None, "The 's2_images' is not set"

    assert isinstance(application_context.s2_images, Sentinel2Images), \
        "Wrong type for 's2_images', expected Sentinel2Images"

    if not configuration.SOURCE_REMOTE:
        assert os.path.exists(application_context.s2_images.path) and \
               os.path.isdir(application_context.s2_images.path), \
            "The path for Sentinel2 images doesn't exist or it is not a folder"
    assert isinstance(application_context.input_parameters.s2_samples_image_name, str) and \
           application_context.input_parameters.s2_samples_image_name is not None and \
           len(application_context.input_parameters.s2_samples_image_name) > 0, \
        "The input parameter for the Sentinel2 image for computing the samples is expected to be a string " \
        "representing a file name"


def get_classificated_image_by_majority_rule(application_context):
    """
        Process the classified images, one for each trial, computed in the STEP 5 of classification.
        By using the majority rule, generate the final prediction map
        :param application_context: The application context
        :type application_context: ApplicationContext
        :return: The final prediction image matrix
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    x_size = application_context.predicted_images[0].shape[0]
    y_size = application_context.predicted_images[0].shape[1]

    classified_images = None
    for predicted_image in application_context.predicted_images:
        reshaped = numpy.reshape(predicted_image, (x_size * y_size, 1))
        if classified_images is None:
            classified_images = reshaped
            continue

        classified_images = numpy.concatenate((classified_images, reshaped), axis=1)

    # Computing the majority rules
    majority_rule = utils.compute_majority_rule(classified_images, True)
    final_image = numpy.reshape(majority_rule, (x_size, y_size, 1))
    return final_image


def get_less_cloudy_feature_selected_matrixes(application_context, less_cloudy_cmasks_names):
    """

        :param application_context: The application context
        :type application_context: ApplicationContext
        :param less_cloudy_cmasks_names: The list of cmasks names ordered by less cloudy to process
        :type less_cloudy_cmasks_names: list of str
        :return: The matrix of s2 images for the corresponding less cloudy images having just the selected features
        :rtype: dict having the key the less_cloudy_cmask_name and as value the corresponding Sentinel2 image already
        filtered by the feature selection
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    result_matrix_map = dict()
    for less_cloudy_name in less_cloudy_cmasks_names:
        bands_to_load = utils.get_feature_selection_by_image_name(application_context, less_cloudy_name)
        s2_image_matrix = resources.get_s2_image_matrix(application_context, less_cloudy_name, bands_to_load)
        result_matrix_map[less_cloudy_name] = s2_image_matrix

    return result_matrix_map


def compute_similiarity_model(application_context, input_matrixes_map, cloudy_image_names, identifier, result_queue):
    """
        Compute the KDTree model of the input_matrixes_map images setting high values for the clouds in the
        cloudy_image_names.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param input_matrixes_map: The map of less cloudy matrixes (Sentinel 2 already feature selected)
        :type input_matrixes_map: dict
        :param cloudy_image_names: The name of the cloudy images (from which remove the cloudy pixels)
        :type cloudy_image_names: list of str
        :param identifier: The execution identifier
        :param result_queue:
        :type result_queue: multiprocessing.Queue
        :return:
    """
    context_information = identifier

    application_context.logger.debug("{} Starting the computation for sub set {}"
                                     .format(context_information, cloudy_image_names))

    x_size = application_context.s2_images.x_size
    y_size = application_context.s2_images.y_size

    # The data to build the KDTree upon
    kdtree_input_data = None
    # The cloudy pixels to rebuild
    cloudy_pixels_indexes_to_rebuild = None
    # The cloudy pixels to exclude (set to 65535) on the kdtree_input_data
    cloudy_pixels_indexes_to_exclude = None

    #
    # Composing the kdtree_input_data by using the elements in the input_matrixes_map which do not appear in the
    # 'cloudy_image_names' (removing completely the features ~ bands of those elements).
    # e.g. 0001 (where 1 in 'cloudy_images_name') -> the kdtree_input_data will have the features ~ bands of three
    # images (corresponding to the 0 ones)
    # Compute the indexes for the cloud pixels to rebuild, which are the ones having at least a cloud.
    #
    for image_name in input_matrixes_map:
        # Retrieve the cloud mask for the image to keep
        cloud_mask = application_context.s2_images.name_cloud_covers_matrix_map[image_name]
        cloud_mask = cloud_mask.reshape(x_size * y_size)
        # In case the cloud mask has 3rd dimension, collapse to 2 for proper further indexing
        if cloud_mask.ndim == 3:
            cloud_mask = cloud_mask[:, :, 0]

        # Process the case the image is in the 'cloudy_image_name' (i.e. 1 ones)
        if image_name in cloudy_image_names:
            if cloudy_pixels_indexes_to_rebuild is None:
                cloudy_pixels_indexes_to_rebuild = cloud_mask > 0
            else:
                cloudy_pixels_indexes_to_rebuild = numpy.logical_or(cloudy_pixels_indexes_to_rebuild, cloud_mask > 0)

            # Nothing more to do, the image has to be discarded
            continue

        # Retrieve the cloudy pixels for the image which will be the input for the KDTree
        if cloudy_pixels_indexes_to_exclude is None:
            cloudy_pixels_indexes_to_exclude = cloud_mask > 0
        else:
            cloudy_pixels_indexes_to_exclude = numpy.logical_or(cloudy_pixels_indexes_to_exclude, cloud_mask > 0)

        # Due feature selection, the 3rd dimension depends on the current execution (varies in size among the images)
        image_matrix = input_matrixes_map[image_name]
        image_matrix = image_matrix.reshape(x_size * y_size, image_matrix.shape[2])

        # Add the image to the model
        if kdtree_input_data is None:
            kdtree_input_data = image_matrix
        else:
            kdtree_input_data = numpy.concatenate((kdtree_input_data, image_matrix), axis=1)

    # e.g. 0001 (where 1 in 'cloudy_image_names') -> set 65535 to cloudy pixels of elements at '0' and remove completely
    # the element at '1'.
    cloudy_pixels_to_rebuild = kdtree_input_data[cloudy_pixels_indexes_to_rebuild]

    # Compute the indexes for the cloud pixels to exclude once the query values have been retrieved.
    type_max_value = numpy.iinfo(kdtree_input_data.dtype).max
    start_band = 0
    for image_name in input_matrixes_map:
        # Skip the ones not used for build the data
        if image_name in cloudy_image_names:
            continue

        # Set just the image cloudy pixels in the build data to 65535
        image_matrix = input_matrixes_map[image_name]
        end_band = start_band + image_matrix.shape[2]
        kdtree_input_data[cloudy_pixels_indexes_to_exclude, start_band:end_band] = type_max_value
        start_band = end_band

    application_context.logger.debug("{} Computing the sampling, for each class, of size {} "
                                     .format(context_information, configuration.KDTREE_MODEL_SAMPLES_NUMBER))
    class_values = numpy.unique(application_context.predicted_image)
    reshaped_predicted_image = application_context.predicted_image.reshape((x_size * y_size))
    sampled_model_input = None
    sampled_model_input_associated_predicted_values = None
    for class_value in class_values:
        # Select the class associated pixels, not being cloudy
        condition = numpy.logical_and(reshaped_predicted_image == class_value,
                                      numpy.max(kdtree_input_data, axis=1) < type_max_value)
        model_input_for_class = kdtree_input_data[condition]
        reshaped_predicted_image_for_class = reshaped_predicted_image[condition]

        # Computing the sample as configured of at most of the elements in the class for the image
        sample_size = numpy.minimum(configuration.KDTREE_MODEL_SAMPLES_NUMBER, model_input_for_class.shape[0])
        model_input_for_class_indexes = \
            utils.get_random_sample(model_input_for_class, sample_size, axis=0, indexes=True)
        model_input_for_class_sampled = model_input_for_class[model_input_for_class_indexes]
        model_input_for_class_sampled_associated_predicted_values = \
            reshaped_predicted_image_for_class[model_input_for_class_indexes]

        if sampled_model_input is None:
            sampled_model_input = model_input_for_class_sampled
            sampled_model_input_associated_predicted_values = model_input_for_class_sampled_associated_predicted_values
            continue

        sampled_model_input = numpy.concatenate((sampled_model_input, model_input_for_class_sampled), axis=0)
        sampled_model_input_associated_predicted_values = \
            numpy.concatenate((sampled_model_input_associated_predicted_values,
                               model_input_for_class_sampled_associated_predicted_values), axis=0)

    application_context.logger.debug("{} Performing KDTree".format(context_information))
    model = cKDTree(sampled_model_input)
    application_context.logger.debug("{} KDTree on the subset successfully computed".format(context_information))

    # Query the model starting from the second nearest neighbour to the configured value
    application_context.logger.debug("{} Querying the KDTree model for {} cloudy pixels with {} neighbours"
                                     .format(context_information, len(cloudy_pixels_to_rebuild),
                                             configuration.CLOUD_REMOVAL_KDTREE_NEIGHBOURS))
    distances_and_neighbours = \
        model.query(cloudy_pixels_to_rebuild, k=range(1, configuration.CLOUD_REMOVAL_KDTREE_NEIGHBOURS))
    neighbours_indexes = distances_and_neighbours[1]

    # Computing the majority rule among the results
    application_context.logger.debug("{} Computing the majority rule among the results".format(context_information))
    neighbours = sampled_model_input_associated_predicted_values[neighbours_indexes]
    majority_ruled_neighbours = utils.compute_majority_rule(neighbours)

    # Saving the majority result for the model in case requested
    if configuration.POST_PROCESS_DEBUG_CLOUDY_PIXELS_RECONSTRUCTION:
        result_to_print = numpy.zeros((x_size * y_size), numpy.uint8)
        result_to_print[cloudy_pixels_indexes_to_rebuild] = majority_ruled_neighbours
        result_to_print = result_to_print.reshape((x_size, y_size))
        destination_file_path = os.path.join(
            application_context.input_parameters.output_path,
            'majority_ruled_neighbours_' + str(identifier) + '.tif')
        utils.save_matrix_as_geotiff(
            result_to_print,
            destination_file_path,
            None,
            configuration.IMAGES_OUTPUT_GDAL_TYPE,
            geo_transform=application_context.s2_images.geo_transform,
            projection=application_context.s2_images.projection
        )

    application_context.logger.debug("{} Computation terminated".format(context_information))
    result_queue.put((identifier, majority_ruled_neighbours, cloudy_pixels_indexes_to_rebuild))


def manage_processes_and_results_models(application_context, result_queue, active_processes, models_results,
                                        models_results_indexes):
    """
        Perform a check on both the result_queue, managing the available results
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param result_queue: The queue hosting the computation results from the terminated processes
        :type result_queue: multiprocessing.queues.Queue
        :param active_processes: The list of active processes (i.e. the ones still computing, but also the one who
        have concluded the computation since last check)
        :type active_processes: list of multiprocessing.Process
        :param models_results: The results dictionary for the models
        :type models_results: dict
        :param models_results_indexes: The indexes for the models results
        :type models_results_indexes: dict
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(result_queue, multiprocessing.queues.Queue), \
        "Wrong type for 'result_queue', expected multiprocessing.queues.Queue"
    assert isinstance(active_processes, list), "Wrong type for 'active_processes', expected list"
    assert all(isinstance(process, multiprocessing.Process) for process in active_processes), \
        "Wrong type for 'active_processes', expected list of multiprocessing.Process"
    assert isinstance(models_results, dict), "Wrong input parameter type for 'models_results', expected dict"

    # Once at least a result is available, acquire it
    while result_queue.qsize() > 0:
        # Removing the result from the queue
        job_result = result_queue.get()
        identifier = job_result[0]
        result_pixels = job_result[1]
        result_pixels_positions = job_result[2]

        application_context.logger.debug("Get a result for the computation {}".format(identifier))
        models_results[identifier] = result_pixels
        models_results_indexes[identifier] = result_pixels_positions

    # Remove the no longer active jobs (using array slicing for avoiding to skip some elements due change of structure)
    for process in active_processes[:]:
        # Skip the process if it is still working
        if process.is_alive():
            continue

        process.join()
        process.close()
        active_processes.remove(process)


def compute_models(application_context, less_cloudy_cmasks_names):
    """
        Compute the models as combination of cloud images (2^|less_cloudy_cmasks_names|).
        Each model will have up to a defined number of samples (e.g. 5000) from the cleared pixels
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param less_cloudy_cmasks_names: The list of cmasks names ordered by less cloudy to process
        :type less_cloudy_cmasks_names: list of str
        :return: The computed models (indexed by the composition of used images names) and the clouds level matrix
        :rtype: dict, numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    # Computing the power set of the cloudy images as model composition reference
    power_set = utils.compute_powerset(less_cloudy_cmasks_names)

    # 20200328 Filtering on the sets having at least one cloud element
    power_set = [s for s in power_set if len(s) > configuration.POST_PROCESSING_MINIMUM_CLOUD_COUNT_MODEL]

    # Filtering out the case of removing all of the clouds for computing the model
    power_set = [element for element in power_set if
                  0 < len(element) < len(less_cloudy_cmasks_names)]

    # Preparing the matrix of S2 images corresponding to the less cloudy cmasks, already filtered by the feature set
    application_context.logger.debug("Preparing the matrix map corresponding to the least cloudy images, having just "
                                     "the selected features")
    input_matrixes_map = get_less_cloudy_feature_selected_matrixes(application_context, less_cloudy_cmasks_names)

    # Multiprocessing
    available_processes = utils.get_processors_count_post_classification()
    application_context.logger.debug("Performing computation on " + str(available_processes) + " processes")
    result_queue = multiprocessing.Queue()
    active_processes = []

    # Process all the sets
    models_results = dict()
    models_results_positions = dict()
    i = 0
    while i < len(power_set):
        sub_set = power_set[i]

        if len(active_processes) < available_processes:
            process = multiprocessing.Process(
                target=compute_similiarity_model,
                args=(application_context, input_matrixes_map, sub_set, i, result_queue))
            active_processes.append(process)
            process.start()
            i += 1
            continue

        # Keep checking while all the 'workers' are still computing
        while len(active_processes) == available_processes:
            application_context.logger.debug("Still no result ...")
            time.sleep(configuration.SVM_THREAD_CHECK_SLEEP)

            # Delegate for saving the result and clearing a terminated process
            manage_processes_and_results_models(application_context, result_queue, active_processes, models_results,
                                                models_results_positions)

    # Once all models have been submitted, wait for termination
    application_context.logger.debug("All models have been submitted. Waiting for computation...")
    while len(active_processes) > 0:
        application_context.logger.debug("Some process is still computing ...")
        time.sleep(configuration.SVM_THREAD_CHECK_SLEEP)
        # Delegate for saving the result and clearing a terminated process
        manage_processes_and_results_models(application_context, result_queue, active_processes, models_results,
                                            models_results_positions)

    application_context.logger.debug("All the models are computed")
    return models_results, models_results_positions


def main(application_context):
    """
        Execute the 'STEP5 Post classification'
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    application_context.logger.info("Starting execution of 'STEP6 Post classification'")
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context)

    # Getting the current tile name and its year
    tile_data = utils.get_tile_date_by_s2_file_name(application_context.input_parameters.s2_samples_image_name)
    tile_name = tile_data[0]
    tile_year = tile_data[1]
    tile_year_name = str(tile_year) + "_" + tile_name

    # Getting the predicted image (majority rule) or compute it
    if application_context.steps_to_skip is None or 61 not in application_context.steps_to_skip:
        application_context.logger.debug("Checking the existence of the final predicted image or load it")
        if not resources.check_or_load_final_predicted_image(application_context, True):
            application_context.logger.debug("Loading the classified images (1 for each trial)")
            resources.check_or_load_predicted_images(application_context)

            application_context.logger.debug("Performing majority rule on the classified images")
            application_context.predicted_image = get_classificated_image_by_majority_rule(application_context)

            application_context.logger.debug("Saving the final predicted image")
            destination_file_path = os.path.join(
                application_context.input_parameters.output_path,
                configuration.CLASSIFICATION_OUTPUT_FINAL_FILE_NAME.format(tile_year_name))
            utils.save_matrix_as_geotiff(
                application_context.predicted_image,
                destination_file_path,
                None,
                configuration.IMAGES_OUTPUT_GDAL_TYPE,
                geo_transform=application_context.s2_images.geo_transform,
                projection=application_context.s2_images.projection,
                colors_maps_array=[configuration.seomClcLegend.get_colors_map()]
            )

            application_context.predicted_images = None # Freeing memory

    if application_context.steps_to_skip is None or 62 not in application_context.steps_to_skip:
        # Finding the less cloudy cmasks images and taking just the required number
        application_context.logger.info("Loading the less cloudy cmasks images")
        resources.check_or_load_cloud_masks(application_context)
        less_cloudy_cmasks_names = utils.get_less_cloudy_cmasks(application_context)
        application_context.logger.debug("The less cloudy images are: " + ",".join(less_cloudy_cmasks_names))

        # Computing the models for all the cloud combinations of the less cloudy images (e.g. 2^4 - 1 -> 15)
        application_context.logger.info("Computing the KDTree models for the combination of the less cloudy images")
        models_results_map, models_results_positions_map = compute_models(application_context, less_cloudy_cmasks_names)

        # Perform majority rule on the results (majority rule of majority rule)
        application_context.logger.info("Composing the predicted image with cloudy pixels replaced by each model")
        x_size = application_context.s2_images.x_size
        y_size = application_context.s2_images.y_size
        cloudy_replaced_prediction_matrixes = None
        for result_key in models_results_map:
            cloudy_pixels_replacement_values = models_results_map[result_key]
            cloudy_pixels_positions = models_results_positions_map[result_key]
            cloudy_pixels_replaced_prediction_matrix = application_context.predicted_image.reshape((x_size * y_size))
            cloudy_pixels_replaced_prediction_matrix[cloudy_pixels_positions] = cloudy_pixels_replacement_values
            if cloudy_replaced_prediction_matrixes is None:
                # Reshaping for avoiding the error "axis 1 is out of bounds for array of dimension 1" which happens on numpy
                # for having two 1D array - internal numpy representation
                cloudy_replaced_prediction_matrixes = cloudy_pixels_replaced_prediction_matrix.reshape(-1, 1)
                continue

            # Reshaping for avoiding the error "axis 1 is out of bounds for array of dimension 1" which happens on numpy
            # for having two 1D array - internal numpy representation
            cloudy_replaced_prediction_matrixes = numpy.concatenate(
                (cloudy_replaced_prediction_matrixes, cloudy_pixels_replaced_prediction_matrix.reshape(-1, 1)), axis=1)

        application_context.logger.info("Performing the majority rule on the matrixes")
        final_predict_image = utils.compute_majority_rule(cloudy_replaced_prediction_matrixes, True)
        final_predict_image = final_predict_image.reshape((x_size, y_size, 1))
        destination_file_path = os.path.join(
            application_context.input_parameters.output_path,
            configuration.POST_PROCESSING_FINAL_IMAGE_NAME.format(tile_year_name))
        application_context.logger.info("Saving the final result " + destination_file_path)
        utils.save_matrix_as_geotiff(
            final_predict_image,
            destination_file_path,
            None,
            configuration.IMAGES_OUTPUT_GDAL_TYPE,
            geo_transform=application_context.s2_images.geo_transform,
            projection=application_context.s2_images.projection,
            colors_maps_array=[configuration.seomClcLegend.get_colors_map()]
        )


if __name__ == "__main__":
    application_context = ApplicationContext()
    output_path = "/mnt/hgfs/shared/clipped/2018_T32TPS/output"
    log_configuration = logging_utils.get_log_configuration("../../configurations/logging-classification.yaml")
    logging_utils.override_log_file_handler_path(log_configuration, 'file', output_path)
    logging_utils.initialize_log(log_configuration)
    application_context.logger = logging_utils.get_logger(configuration.LOGGER_NAME)
    utils.initialize_legend_colors(configuration.seomClcLegend,
                                   os.path.join("../../", configuration.SEOM_COLOR_MAP_PATH))
    application_context.input_parameters.is_to_save_intermediate_outputs = True
    application_context.input_parameters.output_path = output_path
    application_context.input_parameters.s2_data_path = "/mnt/hgfs/shared/clipped/2018_T32TPS/s2_images" # "/mnt/hgfs/seomData/s2/2018_T32TPS/s2_images"
    application_context.input_parameters.s2_samples_image_name = "MSIL2A_20180827T101021_N0208_R022_T32TPS.tif"
    application_context.input_parameters.cloud_mask_image_path = "/mnt/hgfs/shared/clipped/2018_T32TPS/c_masks/Cmask_20180827.tif" # "/mnt/hgfs/seomData/s2/2018_T32TPS/c_masks/Cmask_20180827.tif"
    application_context.s2_images = Sentinel2Images()
    application_context.s2_images.path = application_context.input_parameters.s2_data_path
    application_context.s2_images.name_matrix_map = dict.fromkeys([
        "MSIL2A_20180419T101031_N0207_R022_T32TPS.tif",
        "MSIL2A_20180623T101029_N0206_R022_T32TPS.tif",
        "MSIL2A_20180713T101029_N0206_R022_T32TPS.tif",
        application_context.input_parameters.s2_samples_image_name
        # "MSIL2A_20180926T101021_N0208_R022_T32TPS.tif",
        # "MSIL2A_20181021T101039_N0206_R022_T32TPS.tif"
    ])

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
