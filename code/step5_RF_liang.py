#!/usr/bin/env python
import functools
#import multiprocessing
import numbers
import os
import sys
import time
import traceback
from tqdm import tqdm
import gc
import numpy
import sklearn
from sklearn import svm, model_selection, metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import torch
#from breizhcrops.models import LSTM
from configurations import configuration
from modules import utils, logging_utils
from modules.exceptions.SeomException import SeomException
from modules.models.application_context import ApplicationContext
from modules.models.sentinel2_images import Sentinel2Images
import sklearn.metrics
from torch.multiprocessing import Pool, Queue, Process, set_start_method
from pathlib import Path
#import matplotlib.pyplot as plt
import collections
import argparse

try:
    set_start_method('spawn')
except RuntimeError:
    pass


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


def get_classifier_mean_score(c, gamma, training_labels, training_data):
    """
        Compute the SVM cross-validation score for the given parameters
        :param c: The penalty parameter C
        :type c: Number
        :param gamma: The gamma parameter
        :type gamma: Number
        :param training_labels: The training data labels
        :type training_labels: numpy.ndarray
        :param training_data: The training data
        :type training_data: numpy.ndarray
        :return: (the cross validation score, the c value, the gamma value)
        :rtype: (Number, Number, Number)
    """
    assert isinstance(
        c, numbers.Number), "Wrong type for 'c', expected numbers.Number"
    assert isinstance(
        gamma, numbers.Number), "Wrong type for 'gamma', expected numbers.Number"
    assert isinstance(
        training_labels, numpy.ndarray), "Wrong type for 'training_labels', expected numpy.ndarray"
    assert isinstance(
        training_data, numpy.ndarray), "Wrong type for 'training_data', expected numpy.ndarray"

    # Instantiating the classifier ~ estimator
    classifier = svm.SVC(
        C=c,
        gamma=gamma,
        probability=configuration.SVM_COMPUTE_PROBABILITY)
    cross_validation_scores = model_selection.cross_val_score(
        classifier,
        training_data,
        training_labels,
        cv=configuration.SVM_CROSS_VALIDATION_STRATEGY)
    mean_score = cross_validation_scores.mean() * 100

    return mean_score, c, gamma


def get_best_classifier(
        application_context,
        training_labels,
        training_data,
        svm_c_parameter=None,
        svm_gamma_parameter=None):
    """
        Selecting the best model
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param training_labels: The label of the training data
        :type training_labels: numpy.ndarray
        :param training_data: The training data
        :type training_data: numpy.ndarray
        :param svm_c_parameter: The SVM C parameter to use for building the classifier
        :type svm_c_parameter: float
        :param svm_gamma_parameter: The SVM gamma parameter to use for building the classifier
        :type svm_gamma_parameter: float
        :return: The model to use
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(
        training_labels, numpy.ndarray), "Wrong type for 'training_labels', expected numpy.ndarray"
    assert isinstance(
        training_data, numpy.ndarray), "Wrong type for 'training_data', expected numpy.ndarray"
    assert training_labels.shape[0] == training_data.shape[0], "The parameters have different size for dimension 0"

    logger = application_context.logger

    # In case there is an override from application context, just instantiate
    # the classifier
    if svm_c_parameter is not None and svm_gamma_parameter is not None:
        logger.debug(
            "Using the application context parameters for SVM: c={} and gamma={}" .format(
                str(svm_c_parameter),
                str(svm_gamma_parameter)))
        classifier = svm.SVC(
            C=svm_c_parameter,
            gamma=svm_gamma_parameter,
            probability=configuration.SVM_COMPUTE_PROBABILITY
        ).fit(training_data, training_labels)
        return classifier

    start = time.time()
    # Computing all the permutations for the two arrays (as input of the
    # function to parallelize)
    mesh_grid = numpy.meshgrid(
        configuration.SVM_C_SPACE,
        configuration.SVM_GAMMA_SPACE)
    # Restructuring the permutations to a array of tuples to be accepted by
    # the function
    input_permutations = numpy.array(mesh_grid).T.reshape(-1, 2)
    # Setting the processing pool as the number of the available processes
    processes = Pool(processes=utils.get_processors_count_classification())
    # Setting the common parameters
    partial_function = functools.partial(
        get_classifier_mean_score,
        training_labels=training_labels,
        training_data=training_data)
    # Run the processes
    results = processes.starmap(partial_function, input_permutations)

    # Checking the best values
    best_score = 1
    best_c = None
    best_gamma = None
    for result in results:
        # TODO Seems that the last same score can rewrite the parameters, so are they actually relevant?
        # TODO e.g. All the iterations seems to have the same score for a given C, the gamma doesn't affect the score...
        # TODO Check the most likely couple of values, considering the results
        # may not be in order
        if result[0] >= best_score:
            best_score = result[0]
            best_c = result[1]
            best_gamma = result[2]

    end = time.time()
    logger.info("Execution time " + str(end - start))

    # Instantiate the classifier with the default 'rbf' (Radial Basis
    # Function) and the above computed parameters
    classifier = svm.SVC(
        C=best_c,
        gamma=best_gamma,
        probability=configuration.SVM_COMPUTE_PROBABILITY) .fit(
        training_data,
        training_labels)
    return classifier


def metrics(y_true, y_pred):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(
        y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(
        y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(
        y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(
        y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(
        y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(
        y_true, y_pred, average="weighted")

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )


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

    # In case there is an override from configuration, just compute the target
    # patch
    if configuration.SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_ROW_START is not None and \
            configuration.SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_COLUMN_START is not None:
        patches.append(
            (configuration.SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_ROW_START,
             configuration.SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_ROW_START +
             patch_size,
             configuration.SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_COLUMN_START,
             configuration.SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_COLUMN_START +
             patch_size))
        return patches

    return utils.get_patches_coordinates(x_size, y_size, patch_size)


def classify_time_series_patch(
        result_queue,
        patch_matrix,
        classifier,
        x_start,
        x_end,
        y_start,
        y_end,
        logger):
    """
        Perform a classification of a patch of the time series.
        The result will be set in the result_queue.
        :param result_queue: The result queue, to store the result
        :type result_queue: multiprocessing.Queue
        :param patch_matrix: The matrix representing a portion (patch) of the Sentinel 2 time-series (expected 3D)
        :type patch_matrix: numpy.ndarray
        :param classifier: The classifier to execute
        :type classifier: svm.SVC
        :param x_start: The start index of the patch for the X axis
        :type x_start: int
        :param x_end: The end index of the patch for the X axis
        :type x_end: int
        :param y_start: The start index of the patch for the Y axis
        :type y_start: int
        :param y_end: The end index of the patch for the Y axis
        :type y_end: int
        :param logger: The logger
        :type logger: logging.Logger
    """
    assert isinstance(result_queue, type(Queue())), \
        "Wrong type for 'result_queue', expected multiprocessing.queues.Queue"
    assert isinstance(
        patch_matrix, numpy.ndarray), "Wrong type for 'patch_matrix', expected numpy.ndarray"
    #assert isinstance(classifier, svm.SVC), "Wrong type for 'classifier', expected svm.SVC"
    assert isinstance(x_start, int), "Wrong type for 'x_start', int"
    assert isinstance(x_end, int), "Wrong type for 'x_end', int"
    assert isinstance(y_start, int), "Wrong type for 'y_start', int"
    assert isinstance(y_end, int), "Wrong type for 'y_end', int"

    if x_end <= x_start:
        raise SeomException(
            "The patch coordinates are inconsistent: x_end <= x_start")

    if y_end <= y_start:
        raise SeomException(
            "The patch coordinates are inconsistent: x_end <= x_start")

    if len(patch_matrix.shape) < 3:
        raise SeomException(
            "The patch matrix is expected to be a 3 dimensional matrix")

    if patch_matrix.shape[0] != patch_matrix.shape[1]:
        raise SeomException(
            "The patch matrix is expected to have the same size for the first two dimensions (x and y)")

    # Getting patch dimensions (expected 3D matrix)
    x_size = patch_matrix.shape[0]
    y_size = patch_matrix.shape[1]
    bands_count = patch_matrix.shape[2]

    # Reshaping input
    logger.debug("Reshaping 3D matrix in 2D one for estimator")
    reshaped_patch_matrix = patch_matrix.reshape(x_size * y_size, bands_count)

    # normalize
    reshaped_patch_matrix, num_obs, n_bands = utils.stack_bands(
        reshaped_patch_matrix)
    reshaped_patch_matrix = utils.normalization(reshaped_patch_matrix, -1, 1)
    print("reshaped_patch_matrix min:", reshaped_patch_matrix.min(axis=0))
    print("reshaped_patch_matrix max:", reshaped_patch_matrix.max(axis=0))
    print("reshaped_patch_matrix mean:", reshaped_patch_matrix.mean(axis=0))
    print("reshaped_patch_matrix std:", reshaped_patch_matrix.std(axis=0))
    reshaped_patch_matrix = utils.invert_stack_bands(
        reshaped_patch_matrix, num_obs, n_bands)

    # Prediction
    prediction_result = None
    prediction_result_probabilities = None
    if configuration.SVM_COMPUTE_PROBABILITY:
        logger.debug("Execution of prediction using probabilities")
        # The result of 'predict_proba' has on the dim=2 (columns) the indexes for the element in 'classes_',
        # need to post-process the computation output to the expected form (since there may be some missing classes,
        # e.g. the number 13 'Rice').
        # It is a matrix having rows as the elements and columns as the number
        # of the classes in the classifier
        prediction = classifier.predict_proba(reshaped_patch_matrix)
        # Getting the classes indexes having the maximum probability for each
        # element
        prediction_class_indices = prediction.argmax(axis=1)

        # The prediction result likewise the one computed with 'predict'
        prediction_result = numpy.zeros(
            (x_size * y_size),
            dtype=configuration.IMAGES_CLC_NUMPY_TYPE)
        # The prediction probabilities result
        prediction_result_probabilities = numpy.zeros(
            (x_size * y_size, len(configuration.seomClcLegend.seom_classes)),
            dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)

        # Reconstructing the actual results by reverse mapping
        for i in range(0, len(classifier.classes_)):
            # Getting the actual class code (label)
            class_value = classifier.classes_[i]
            # Setting the results for the elements having the result index in
            # the current position
            prediction_result[prediction_class_indices == i] = class_value
            # Setting correctly the probabilities through reverse mapping
            prediction_result_probabilities[:,
                                            class_value - 1] = prediction[:, i]
    else:
        logger.debug("Execution of prediction")
        #prediction_result = classifier.predict(reshaped_patch_matrix)
        if isinstance(classifier, RandomForestClassifier):
            prediction_result = classifier.predict(reshaped_patch_matrix)
        else:
            prediction_result, _ = test_lstm(
                logger, reshaped_patch_matrix, classifier)

    # Reshaping result
    logger.debug("Reshaping resulting 1D matrix to 2D one for result")
    reshaped_patch_matrix = prediction_result.reshape(x_size, y_size)
    print("reshaped_patch_matrix", reshaped_patch_matrix)
    if prediction_result_probabilities is not None:
        prediction_result_probabilities = prediction_result_probabilities.reshape(
            x_size, y_size, prediction_result_probabilities.shape[1])
    #result_queue.put((reshaped_patch_matrix, x_start, x_end, y_start, y_end, prediction_result_probabilities))
    return reshaped_patch_matrix


def write_prediction_probabilities(
        application_context,
        prediction_result_probabilities,
        execution_id,
        row_offset=0,
        column_offset=0):
    """
        Write the classification prediction probabilities for the classes in an external file (due the huge memory size
        requirement)
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param prediction_result_probabilities: The matrix having the probabilities for each class (e.g. 2196x2196x16)
        :type prediction_result_probabilities: numpy.ndarray
        :param execution_id: An identifier of the current running context (it will part of the file name since it is
        related to the running trial)
        :type execution_id: int
        :param row_offset: The number of rows to skip before the write (0 means start of the file-rows)
        :type row_offset: int
        :param column_offset: The number of columns to skip before the write (0 means start of the file-columns)
        :type column_offset: int
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(prediction_result_probabilities, numpy.ndarray), \
        "Wrong input parameter type for 'prediction_result_probabilities', expected numpy.ndarray"
    assert isinstance(
        execution_id, int), "Wrong type for 'execution_id', expected int"
    assert isinstance(
        row_offset, int), "Wrong type for 'row_offset', expected int"
    assert isinstance(
        column_offset, int), "Wrong type for 'column_offset', expected int"

    destination_file_path = os.path.join(
        application_context.input_parameters.output_path,
        configuration.CLASSIFICATION_PROBABILITIES_OUTPUT_FILE_NAME.format(execution_id))

    # Create the file which will hosts the probabilities (e.g. 10980x10980xn)
    if not os.path.exists(destination_file_path):
        application_context.logger.debug(
            "Generating prediction probabilities file in " +
            destination_file_path)
        x_size = application_context.s2_images.x_size
        y_size = application_context.s2_images.y_size
        bands = len(configuration.seomClcLegend.seom_classes)
        basic_matrix = numpy.zeros(
            (x_size, y_size, bands), dtype=configuration.IMAGES_NORMALIZED_NUMPY_TYPE)
        utils.save_matrix_as_geotiff(
            basic_matrix,
            destination_file_path,
            None,
            configuration.IMAGES_PROBABILITIES_GDAL_TYPE,
            geo_transform=application_context.s2_images.geo_transform,
            projection=application_context.s2_images.projection,
            override_output=True
        )

    # Updating the file
    application_context.logger.debug(
        "Updating prediction probabilities in " +
        destination_file_path +
        " for offset "
        "rows=" +
        str(row_offset) +
        " columns=" +
        str(column_offset))
    utils.save_matrix_as_geotiff(
        prediction_result_probabilities,
        destination_file_path,
        None,
        configuration.IMAGES_PROBABILITIES_GDAL_TYPE,
        geo_transform=application_context.s2_images.geo_transform,
        projection=application_context.s2_images.projection,
        row_offset=row_offset,
        column_offset=column_offset,
        override_output=False
    )
    application_context.logger.debug(
        "Prediction probabilities successfully saved")


def manage_processes_and_results(
        application_context,
        result_queue,
        active_processes,
        result_matrix,
        execution_id):
    """
        Perform a check on both the result_queue, managing the available results to the result_matrix, and the
        active_processing, removing the ones which terminated.
        Moreover, in case there are data on the classification classes probabilities, it will update the additional
        output directly; that's because the result matrix will be huge (e.g. 10980*10980*n, where n is the number of the
        classes)
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param result_queue: The queue hosting the computation results from the terminated processes
        :type result_queue: multiprocessing.queues.Queue
        :param active_processes: The list of active processes (i.e. the ones still computing, but also the one who
        have concluded the computation since last check)
        :type active_processes: list of multiprocessing.Process
        :param result_matrix: The matrix hosting the full result of the computation (e.g. the 10980x10980x1).
        Its values are among 1..n included where n is the highest SEOM class value from the legend
        :type result_matrix: numpy.ndarray
        :param execution_id: An identifier of the current running context
        :type execution_id: int
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    assert isinstance(result_queue, type(Queue())), \
        "Wrong type for 'result_queue', expected multiprocessing.Queue"
    assert isinstance(
        active_processes, list), "Wrong type for 'active_processes', expected list"
    assert all(isinstance(process, type(Process())) for process in active_processes), \
        "Wrong type for 'active_processes', expected list of multiprocessing.Process"
    assert isinstance(
        result_matrix, numpy.ndarray), "Wrong type for 'result_matrix', expected numpy.ndarray"
    assert isinstance(
        execution_id, int), "Wrong type for 'execution_id', expected int"

    # Once at least a result is available, acquire it
    while result_queue.qsize() > 0:
        # Removing the result from the queue
        job_result = result_queue.get()
        values = job_result[0]
        print("values:", values)
        x_start = job_result[1]
        x_end = job_result[2]
        y_start = job_result[3]
        y_end = job_result[4]
        prediction_result_probabilities = job_result[5]
        application_context.logger.debug(
            "Get a result for the patch {}:{} {}:{}".format(
                x_start, x_end, y_start, y_end))
        result_matrix[x_start:x_end, y_start:y_end] = values
        print("result_matrix shape", result_matrix.shape)
        print("result_matrix:", result_matrix)

        if configuration.SAVE_PREDICTION_PROBABILITIES and prediction_result_probabilities is not None:
            write_prediction_probabilities(
                application_context,
                prediction_result_probabilities,
                execution_id,
                row_offset=x_start,
                column_offset=y_start)

    # Remove the no longer active jobs (using array slicing for avoiding to
    # skip some elements due change of structure)
    for process in active_processes[:]:
        # Skip the process if it is still working
        if process.is_alive():
            continue

        process.join()
        process.close()
        active_processes.remove(process)


def classify_time_series(
        application_context,
        classifier,
        training_labels,
        training_data,
        min_values,
        max_values,
        execution_id):
    """
        Consider the Sentinel2 time-series and apply the classifier on those images.
        :param application_context: The application context
        :type application_context: ApplicationContext
        :param classifier: The classifier to use (expected to be a SVM one)
        :type classifier: svm.SVC
        :param training_labels: The training labels
        :type training_labels: numpy.ndarray
        :param training_data: The training data
        :type training_data: numpy.ndarray
        :param min_values: Minimum values computed for the columns of the training data (i.e. the features ~ bands)
        :type min_values: numpy.ndarray
        :param max_values: Maximum values computed for the columns of the training data (i.e. the features ~ bands)
        :type max_values: numpy.ndarray
        :param execution_id: An identifier of the current running context
        :type execution_id: int
        :return: The predicted map
        :rtype: numpy.ndarray
    """
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"
    """
    assert isinstance(classifier, svm.SVC), \
        "Wrong input parameter type for 'classifier', expected svm.SVC"
    """
    assert isinstance(training_labels, numpy.ndarray), \
        "Wrong input parameter type for 'training_labels', expected numpy.ndarray"
    assert isinstance(training_data, numpy.ndarray), \
        "Wrong input parameter type for 'training_data', expected numpy.ndarray"
    assert isinstance(
        min_values, numpy.ndarray), "Wrong input parameter type for 'min_values', expected numpy.ndarray"
    assert isinstance(
        max_values, numpy.ndarray), "Wrong input parameter type for 'max_values', expected numpy.ndarray"
    assert isinstance(
        execution_id, int), "Wrong input parameter type for 'execution_id', expected int"

    # Load Sentinel2 images into memory
    application_context.logger.debug(
        "Retrieving the Sentinel2 time series into memory...")
    s2_name_matrix_map = application_context.s2_images.name_matrix_map
    for s2_name in s2_name_matrix_map:
        if s2_name_matrix_map[s2_name] is None:
            path = os.path.join(application_context.s2_images.path, s2_name)
            application_context.logger.debug("Loading the image " + path)
            s2_name_matrix_map[s2_name] = utils.retrieve_image_matrix(
                path,
                target_type=configuration.IMAGES_S2_NUMPY_TYPE)
            #s2_name_matrix_map[s2_name] = utils.get_normalized_image_matrix2d_percentile(s2_name_matrix_map[s2_name])

    """
    with open('s2mat.npy', 'wb') as f:
            numpy.save(f, s2_name_matrix_map)
    """
    # Getting a Sentinel2 image for retrieving the sizes
    x_size = application_context.s2_images.x_size
    y_size = application_context.s2_images.y_size
    bands_size = application_context.s2_images.bands_size

    # Getting the patch size and the patches to process
    patch_size = configuration.SVM_CLASSIFICATION_TIME_SERIES_PATCH_SIZE
    application_context.logger.debug(
        "Computing the patches 'coordinates' of size " +
        str(patch_size))
    patches = get_patches_coordinates(x_size, y_size, patch_size)

    # Multiprocessing
    #available_processes = utils.get_processors_count_classification()
    #print("available_processes", available_processes)
    available_processes = 1
    application_context.logger.debug(
        "Performing computation on " +
        str(available_processes) +
        " processes")
    result_queue = Queue()
    active_processes = []
    """
    #load, clip and normalize images to be predicted
    if configuration.DEV_ENVIRONMENT:
        application_context.logger.debug("[DEV_ENVIRONMENT] Retrieving the Sentinel2 time series into memory...")
        s2_name_matrix_map = application_context.s2_images.name_matrix_map
        for s2_name in s2_name_matrix_map:
            if s2_name_matrix_map[s2_name] is None:
                path = os.path.join(application_context.s2_images.path, s2_name)
                application_context.logger.debug("Loading the image " + path)
                s2_name_matrix_map[s2_name] = utils.retrieve_image_matrix(
                    path,
                    target_type=configuration.IMAGES_S2_NUMPY_TYPE)
                application_context.logger.debug("[DEV_ENVIRONMENT] Normalizing the image...")
                s2_name_matrix_map[s2_name] = utils.get_normalized_image_matrix2d_percentile(s2_name_matrix_map[s2_name])
    """
    result_matrix = numpy.zeros(
        (x_size, y_size), dtype=configuration.IMAGES_S2_NUMPY_TYPE)
    i = 0
    while i < len(patches):
        """
        if configuration.DEV_ENVIRONMENT:
            application_context.logger.debug("[DEV_ENVIRONMENT] Retrieving the Sentinel2 time series into memory...")
            s2_name_matrix_map = application_context.s2_images.name_matrix_map
            for s2_name in s2_name_matrix_map:
                if s2_name_matrix_map[s2_name] is None:
                    path = os.path.join(application_context.s2_images.path, s2_name)
                    application_context.logger.debug("Loading the image " + path)
                    s2_name_matrix_map[s2_name] = utils.retrieve_image_matrix(
                        path,
                        target_type=configuration.IMAGES_S2_NUMPY_TYPE)
        """

        patch = patches[i]
        application_context.logger.debug("Processing patch " + str(patch))

        # Populate the patch_matrix
        # The matrix hosting the patch for all the Sentinel2 time series (so same image portion for all the images).
        # It will be x_size * y_size * (bands_size * len(s2_name_matrix_map)
        patch_matrix = numpy.zeros(
            (patch_size,
             patch_size,
             bands_size *
             len(s2_name_matrix_map)),
            dtype=configuration.IMAGES_S2_NUMPY_TYPE)
        start_band = 0
        x_start = patch[0]
        x_end = patch[1]
        y_start = patch[2]
        y_end = patch[3]
        application_context.logger.debug(
            "Preparing the matrix with the Sentinel time-series for area {}:{} {}:{}" .format(
                x_start, x_end, y_start, y_end))
        for s2_name in s2_name_matrix_map:
            end_band = start_band + bands_size
            s2_image_matrix = s2_name_matrix_map[s2_name]
            #s2_image_matrix = utils.get_normalized_image_matrix2d_percentile(s2_image_matrix)
            patch_matrix[:,
                         :,
                         start_band:end_band] = s2_image_matrix[x_start:x_end,
                                                                y_start:y_end,
                                                                0:bands_size]
            start_band = end_band

        """
        #patch_matrix = patch_matrix.astype(numpy.float32)
        print("patch_matrix shape:", patch_matrix.shape)
        print("patch_matrix mean:", patch_matrix.mean(axis=(0,1)))
        print("patch_matrix mean:", patch_matrix.std(axis=(0,1)))
        """
        # Feature selection
        application_context.logger.debug(
            "Performing feature selection on the Sentinel2 data")
        # remove SVM feature set
        #patch_matrix = patch_matrix[:, :, configuration.SVM_FEATURE_SET]

        # Normalizing theto
        application_context.logger.debug("Normalizing patch data")
        #patch_matrix = utils.get_normalized_image_matrix(patch_matrix, min_values, max_values)
        # Need to set both lower bound to 0 and upper bound to 1 (since the normalization data come from training set
        # there could be some values which aren't being properly normalized)
        #patch_matrix[patch_matrix < 0] = 0
        #patch_matrix[patch_matrix > 1] = 1

        if configuration.DEV_ENVIRONMENT:
            application_context.logger.debug(
                "[DEV_ENVIRONMENT] Removing all the images in the s2_name_matrix_map")
            for s2_name in s2_name_matrix_map:
                s2_name_matrix_map[s2_name] = None
            gc.collect()

        # In case there could be additional processes, initialize them and go
        # to the next patch
        if len(active_processes) < available_processes:
            application_context.logger.debug("Starting the classification")
            # Cloning the classifier (for multiprocessing, each computation a classifier for its lack of thread safety)
            #new_classifier = sklearn.base.clone(classifier)
            #new_classifier = new_classifier.fit(training_data, training_labels)
            #new_classifier = classifier
            """
            process = Process(
                target=classify_time_series_patch,
                args=(result_queue, patch_matrix, classifier, x_start, x_end, y_start, y_end,
                      application_context.logger))
            active_processes.append(process)
            process.start()
            """
            result_patch_matrix = classify_time_series_patch(
                result_queue,
                patch_matrix,
                classifier,
                x_start,
                x_end,
                y_start,
                y_end,
                application_context.logger)
            print("result_patch_matrix", result_patch_matrix)
            result_matrix[x_start:x_end, y_start:y_end] = result_patch_matrix
            i += 1
            continue

        # Keep checking while all the 'workers' are still computing
        while len(active_processes) == available_processes:
            application_context.logger.debug("Still no result ...")
            time.sleep(configuration.SVM_THREAD_CHECK_SLEEP)

            # Delegate for saving the result and clearing a terminated process
            manage_processes_and_results(
                application_context,
                result_queue,
                active_processes,
                result_matrix,
                execution_id)

    # Once all patch have been submitted, wait for termination
    application_context.logger.debug(
        "All patches have been submitted. Waiting for computation...")
    while len(active_processes) > 0:
        application_context.logger.debug("Some process is still computing ...")
        time.sleep(configuration.SVM_THREAD_CHECK_SLEEP)
        # Delegate for saving the result and clearing a terminated process
        manage_processes_and_results(
            application_context,
            result_queue,
            active_processes,
            result_matrix,
            execution_id)

    return result_matrix


def main(application_context):
    """
        Train LSTM and predict patches
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    application_context.logger.info(
        "Starting execution of 'STEP5 SVM classification'")
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context)

    application_context.logger.debug(
        "Some parameters of the upcoming processing:")
    application_context.logger.debug(
        "\tThe feature set: " + str(configuration.SVM_FEATURE_SET))
    application_context.logger.debug("\tThe cross validation strategy: " +
                                     str(configuration.SVM_CROSS_VALIDATION_STRATEGY))
    application_context.logger.debug(
        "\tThe C-space: " + str(configuration.SVM_C_SPACE))
    application_context.logger.debug(
        "\tThe Gamma-space: " + str(configuration.SVM_GAMMA_SPACE))

    for i in range(0, 1):
        training_set_id = i + 1
        #application_context.logger.info("Computing the training set " + str(training_set_id))
        application_context.logger.info("Loading entire training set")
        training_set = application_context.training_set
        training_labels = training_set[:, 0]
        training_data = training_set[:, 1:]

        #actual_training_data = training_data[:, configuration.SVM_FEATURE_SET]
        actual_training_data = training_data
        # Compute the minimum along each column (if required)
        if application_context.min_values is not None:
            application_context.logger.debug(
                "Using application context min_values")
            min_values = application_context.min_values[i]
        else:
            min_values = numpy.amin(actual_training_data, axis=0)
        # Compute the maximum along each column (if required)
        if application_context.max_values is not None:
            application_context.logger.debug(
                "Using application context max_values")
            max_values = application_context.max_values[i]
        else:
            max_values = numpy.amax(actual_training_data, axis=0)

        # Normalizing the data
        #actual_training_data = utils.get_normalized_image_matrix(actual_training_data, min_values, max_values)
        # normalize observations alltogether
        actual_training_data, num_obs, n_bands = utils.stack_bands(
            actual_training_data)
        #actual_training_data = utils.get_normalized_image_matrix_percentile(actual_training_data)
        actual_training_data = utils.normalization(actual_training_data, -1, 1)
        print("actual_training_data min:", actual_training_data.min(axis=0))
        print("actual_training_data max:", actual_training_data.max(axis=0))
        print("actual_training_data mean:", actual_training_data.mean(axis=0))
        print("actual_training_data std:", actual_training_data.std(axis=0))
        actual_training_data = utils.invert_stack_bands(
            actual_training_data, num_obs, n_bands)
        application_context.logger.debug(
            "Setting up LSTM, model will train on the training set " +
            str(training_set_id))

        print("training labels min: ", training_labels.min())
        print("training labels max: ", training_labels.max())

        #classifier = train_lstm(application_context, training_labels, actual_training_data)

        file_path = output_path+"trainingSet3.csv"
        val_set = numpy.loadtxt(
            file_path,
            delimiter=",",
            dtype=configuration.IMAGES_S2_NUMPY_TYPE)
        val_labels = training_set[:, 0]
        val_data = training_set[:, 1:]
        val_data, num_obs, n_bands = utils.stack_bands(val_data)
        val_data = utils.normalization(val_data, -1, 1)
        print("val_data min:", val_data.min(axis=0))
        print("val_data max:", val_data.max(axis=0))
        print("val_data mean:", val_data.mean(axis=0))
        print("val_data std:", val_data.std(axis=0))
        val_data = utils.invert_stack_bands(val_data, num_obs, n_bands)
        classifier = train_val_lstm(
            application_context,
            training_labels,
            actual_training_data,
            val_labels,
            val_data)

        application_context.logger.info(
            "Starting the classification of the Sentinel 2 time series")
        result_matrix = classify_time_series(
            application_context,
            classifier,
            training_labels,
            actual_training_data,
            min_values,
            max_values,
            training_set_id
        )
        destination_file_path = os.path.join(
            application_context.input_parameters.output_path,
            configuration.CLASSIFICATION_OUTPUT_FILE_NAME.format(training_set_id))
        application_context.logger.info(
            "Saving predicted image for {} in {}".format(
                training_set_id, destination_file_path))
        # add 1 that was substracted before training
        result_matrix = result_matrix + 1
        print("result_matrix", result_matrix)
        # Save the result
        utils.save_matrix_as_geotiff(
            result_matrix,
            destination_file_path,
            None,
            configuration.IMAGES_OUTPUT_GDAL_TYPE,
            geo_transform=application_context.s2_images.geo_transform,
            projection=application_context.s2_images.projection,
            colors_maps_array=[configuration.seomClcLegend.get_colors_map()]
        )
        result_matrix = None  # Freeing memory

    application_context.logger.info(
        "Execution of 'STEP5 SVM classification' successfully completed")
    return None


def train_validate_TS(application_context):
    """
        Train on TS of 2018 and validate on TS of 2019
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    application_context.logger.info(
        "Starting execution of 'STEP5 SVM classification'")
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context)

    application_context.logger.debug(
        "Some parameters of the upcoming processing:")
    application_context.logger.debug(
        "\tThe feature set: " + str(configuration.SVM_FEATURE_SET))
    application_context.logger.debug("\tThe cross validation strategy: " +
                                     str(configuration.SVM_CROSS_VALIDATION_STRATEGY))
    application_context.logger.debug(
        "\tThe C-space: " + str(configuration.SVM_C_SPACE))
    application_context.logger.debug(
        "\tThe Gamma-space: " + str(configuration.SVM_GAMMA_SPACE))

    for i in range(0, len(application_context.training_sets)):
        training_set_id = i + 1
        application_context.logger.info(
            "Computing the training set " +
            str(training_set_id))
        training_set = application_context.training_sets[i]
        training_labels = training_set[:, 0]
        training_data = training_set[:, 1:]

        #actual_training_data = training_data[:, configuration.SVM_FEATURE_SET]
        actual_training_data = training_data
        # Compute the minimum along each column (if required)
        if application_context.min_values is not None:
            application_context.logger.debug(
                "Using application context min_values")
            min_values = application_context.min_values[i]
        else:
            min_values = numpy.amin(actual_training_data, axis=0)
        # Compute the maximum along each column (if required)
        if application_context.max_values is not None:
            application_context.logger.debug(
                "Using application context max_values")
            max_values = application_context.max_values[i]
        else:
            max_values = numpy.amax(actual_training_data, axis=0)

        # Normalizing the data
        #actual_training_data = utils.get_normalized_image_matrix(actual_training_data, min_values, max_values)
        # normalize observations alltogether
        actual_training_data, num_obs, n_bands = utils.stack_bands(
            actual_training_data)
        #actual_training_data = utils.get_normalized_image_matrix_percentile(actual_training_data)
        actual_training_data = utils.normalization(actual_training_data, -1, 1)
        print("actual_training_data min:", actual_training_data.min(axis=0))
        print("actual_training_data max:", actual_training_data.max(axis=0))
        print("actual_training_data mean:", actual_training_data.mean(axis=0))
        print("actual_training_data std:", actual_training_data.std(axis=0))
        actual_training_data = utils.invert_stack_bands(
            actual_training_data, num_obs, n_bands)
        application_context.logger.debug(
            "Setting up LSTM, model will train on the training set " +
            str(training_set_id))

        print("training labels min: ", training_labels.min())
        print("training labels max: ", training_labels.max())
        file_path = "/p/project/joaiml/liang1/data/output4/STEP4/data/trainingset_noharm_noorbit/2018/trainingSet3.csv"
        val_set = numpy.loadtxt(
            file_path,
            delimiter=",",
            dtype=configuration.IMAGES_S2_NUMPY_TYPE)
        val_labels = training_set[:, 0]
        val_data = training_set[:, 1:]
        val_data, num_obs, n_bands = utils.stack_bands(val_data)
        val_data = utils.normalization(val_data, -1, 1)
        print("val_data min:", val_data.min(axis=0))
        print("val_data max:", val_data.max(axis=0))
        print("val_data mean:", val_data.mean(axis=0))
        print("val_data std:", val_data.std(axis=0))
        val_data = utils.invert_stack_bands(val_data, num_obs, n_bands)
        classifier = train_val_lstm(
            application_context,
            training_labels,
            actual_training_data,
            val_labels,
            val_data)
    application_context.logger.info(
        "Execution of training and validation on 2018 and 2019 TS successfully completed")
    return None


def train_pred_RF(application_context):
    """
        Execute the 'STEP5 RF classification'
        :param application_context: The application context
        :type application_context: ApplicationContext
    """
    application_context.logger.info(
        "Starting execution of 'STEP5 RF classification'")
    assert isinstance(application_context, ApplicationContext), \
        "Wrong input parameter type for 'application_context', expected ApplicationContext"

    application_context.logger.debug("Checking input parameters")
    validate_step_data(application_context)

    application_context.logger.debug(
        "Some parameters of the upcoming processing:")
    application_context.logger.debug(
        "\tThe feature set: " + str(configuration.SVM_FEATURE_SET))
    application_context.logger.debug("\tThe cross validation strategy: " +
                                     str(configuration.SVM_CROSS_VALIDATION_STRATEGY))
    application_context.logger.debug(
        "\tThe C-space: " + str(configuration.SVM_C_SPACE))
    application_context.logger.debug(
        "\tThe Gamma-space: " + str(configuration.SVM_GAMMA_SPACE))

    for i in range(0, 1):
        training_set_id = i + 1
        application_context.logger.info(
            "Computing the training set " +
            str(training_set_id))
        training_set = application_context.training_sets[i]
        training_labels = training_set[:, 0]

        # select columns corresponding to correct months
        # modifying training set loading
        training_data = training_set[:, 1:]        
        print("training_data shape", training_data.shape)
        actual_training_data = training_data
        # Compute the minimum along each column (if required)
        if application_context.min_values is not None:
            application_context.logger.debug(
                "Using application context min_values")
            min_values = application_context.min_values[i]
        else:
            min_values = numpy.amin(actual_training_data, axis=0)
        # Compute the maximum along each column (if required)
        if application_context.max_values is not None:
            application_context.logger.debug(
                "Using application context max_values")
            max_values = application_context.max_values[i]
        else:
            max_values = numpy.amax(actual_training_data, axis=0)

        # Normalizing the data
        #actual_training_data = utils.get_normalized_image_matrix(actual_training_data, min_values, max_values)
        # normalize observations alltogether
        actual_training_data, num_obs, n_bands = utils.stack_bands(
            actual_training_data)
        #actual_training_data = utils.get_normalized_image_matrix_percentile(actual_training_data)
        actual_training_data = utils.normalization(actual_training_data, -1, 1)
        print("actual_training_data min:", actual_training_data.min(axis=0))
        print("actual_training_data max:", actual_training_data.max(axis=0))
        print("actual_training_data mean:", actual_training_data.mean(axis=0))
        print("actual_training_data std:", actual_training_data.std(axis=0))
        actual_training_data = utils.invert_stack_bands(
            actual_training_data, num_obs, n_bands)
        application_context.logger.debug(
            "Setting up RF, model will train on the training set " +
            str(training_set_id))

        print("training labels min: ", training_labels.min())
        print("training labels max: ", training_labels.max())

        model = RandomForestClassifier()
        """
        # evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, actual_training_data, training_labels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        # report performance
        print('Accuracy: %.3f (%.3f)' % (numpy.mean(n_scores), numpy.std(n_scores)))
        """
        model.fit(actual_training_data, training_labels)

        application_context.logger.info(
            "Starting the classification of the Sentinel 2 time series")
        result_matrix = classify_time_series(
            application_context,
            model,
            training_labels,
            actual_training_data,
            min_values,
            max_values,
            training_set_id
        )
        destination_file_path = os.path.join(
            application_context.input_parameters.output_path,
            configuration.CLASSIFICATION_RF_OUTPUT_FILE_NAME.format(application_context.id_acquisition))
        application_context.logger.info(
            "Saving predicted image for {} in {}".format(
                training_set_id, destination_file_path))

        print("result_matrix", result_matrix)
        # Save the result
        utils.save_matrix_as_geotiff(
            result_matrix,
            destination_file_path,
            None,
            configuration.IMAGES_OUTPUT_GDAL_TYPE,
            geo_transform=application_context.s2_images.geo_transform,
            projection=application_context.s2_images.projection,
            colors_maps_array=[configuration.seomClcLegend.get_colors_map()]
        )
        result_matrix = None  # Freeing memory

    application_context.logger.info("RF crossval finished")
    return None


if __name__ == "__main__":
    application_context = ApplicationContext()
    
    #set the task id
    parser = argparse.ArgumentParser(description='Setting argument parser')
    parser.add_argument("--id_acquisition", help="Select id of the acquisition for s5", type=int, default=1)
    args = parser.parse_args()
    application_context.id_acquisition = args.id_acquisition

    year = "2018"
    path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/input/"
    #tile_list = ['31UES', '31UET', '31UFS', '31UFT', '31UFU', '31UFV', '31UGS', '31UGT', '31UGU', '31UGV']
    tile_list = ['31UFS', '31UFT']
    for i in range(len(tile_list)):
        if application_context.id_acquisition == i:            
            tile = tile_list[i]
            application_context.input_parameters.s2_data_path = path + tile + "/composite/" + year + "/"
            #output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output_12/" + tile + "/" + year + "/"
            output_path = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/new/output/test" + tile + "/" + year
    application_context.input_parameters.s2_samples_image_name = "2018_season1.tif"
    
    log_configuration = logging_utils.get_log_configuration(os.path.join(
        "./", configuration.LOGGING_CLASSIFICATION_CONFIGURATION_FILE_PATH))
    logging_utils.override_log_file_handler_path(
        log_configuration, 'file', output_path)
    logging_utils.initialize_log(log_configuration)
    application_context.logger = logging_utils.get_logger(
        configuration.LOGGER_NAME)
    utils.initialize_legend_colors(
        configuration.seomClcLegend,
        configuration.SEOM_COLOR_MAP_PATH)
    application_context.input_parameters.is_to_save_intermediate_outputs = True    
    application_context.input_parameters.output_path = output_path
    application_context.s2_images = Sentinel2Images()
    application_context.s2_images.path = application_context.input_parameters.s2_data_path
    application_context.s2_images.name_matrix_map = dict.fromkeys(
        ['2018_season1.tif',
         '2018_season2.tif', 
         '2018_season3.tif'])    

    application_context.training_sets = []
    '''
    for trial in range(0, 2):
        file_path = os.path.join(
            application_context.input_parameters.output_path,
            configuration.TRIALS_OUTPUT_FILE_NAME.format(trial + 1))
        training_set = numpy.loadtxt(
            file_path,
            delimiter=",",
            dtype=configuration.IMAGES_S2_NUMPY_TYPE)
        application_context.training_sets.append(training_set)

    application_context.training_set = numpy.vstack(
        application_context.training_sets)
        '''
    # define the path of the merged training set
    path_training = "/p/project/sdlrs/tian1/lcmap_Netherlands/data/output_12/"
    for trial in range(0, 2):
        file_path = os.path.join(
            path_training,
            configuration.TRIALS_OUTPUT_FILE_NAME.format(trial + 1))
        training_set = numpy.loadtxt(
            file_path,
            delimiter=",",
            dtype=configuration.IMAGES_S2_NUMPY_TYPE)
        application_context.training_sets.append(training_set)

    application_context.training_set = numpy.vstack(
        application_context.training_sets)

    try:
        path = os.path.join(
            application_context.s2_images.path,
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
        raise SeomException(
            "Unable to retrieve the Sentinel2 samples image", e)

    try:
        train_pred_RF(application_context)
    except Exception as e:
        if application_context is not None and application_context.logger is not None:
            application_context.logger.critical(traceback.format_exc())
        else:
            print(str(traceback.format_exc()))
