import os
import random

import yaml
import datetime
import logging
import logging.config

from modules.custom_logger import CustomLogger
from modules.exceptions.SeomException import SeomException


def get_log_configuration(log_configuration_file_path):
    """
        Retrieve the log configuration given the log configuration file path
        :param log_configuration_file_path: The logging configuration file
        :type log_configuration_file_path: str
        :return: The log configuration
        :rtype: dict
    """
    # check existence of log configuration file
    absolute_path = os.path.realpath(os.path.expanduser(log_configuration_file_path))
    if not os.path.exists(absolute_path) or \
            not os.path.isfile(absolute_path):
        raise SeomException("The configured log configuration file path is missing or wrong")

    # Acquiring the configuration from YAML
    log_configuration = None
    with open(absolute_path) as f:
        log_configuration = yaml.safe_load(f)

    return log_configuration


def get_logger(logger_name):
    """
        Retrieve the given logger using the application configuration logging
        :param logger_name: The name of the configured logger (in the logging configuration file) to use
        :type logger_name: str
        :return: The requested logger
        :type: logging.Logger
    """
    assert isinstance(logger_name, str), "Wrong input parameter type for 'logger_name', expected str"
    assert logger_name in logging.root.manager.loggerDict, \
        "The requested logger is not available, maybe the logging has not been properly initialized"

    # Saving the original logger to restore later
    logging_class = logging.getLoggerClass()

    # Lock for thread safety
    logging._acquireLock()

    try:
        # Temporary specify the logger factory for the custom logger
        logging.setLoggerClass(CustomLogger)

        # Retrieving the logger from configuration (NOTE: this will be already a logger of logging_class
        source_logger = logging.getLogger(logger_name)

        # Instantiate the custom logger
        logger = logging.getLogger("custom_" + logger_name)

        # Removing default handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)

        # Acquiring from source logger the relevant configuration
        logger.setLevel(source_logger.level)
        for handler in source_logger.handlers:
            logger.addHandler(handler)

        # Restore the original logger
        logging.setLoggerClass(logging_class)

        return logger
    finally:
        logging._releaseLock()


def initialize_log(log_configuration):
    """
        Initialize the log for the application using the logging module
        :param log_configuration: The log configuration to use for initialize the logging for the application
        :type log_configuration: dict
    """
    assert isinstance(log_configuration, dict), "Wrong input parameter type for 'log_configuration', expected dict"
    logging.config.dictConfig(log_configuration)


def override_log_file_handler_path(log_configuration, handler_name='file', output_path_prefix=None):
    """
        Override the log file handler path for the given handler in the configuration.
        It is used for setting the date and time in the file name and, in case it is given, the folder position.
        :param log_configuration: The log configuration to use for initialize the logging for the application
        :type log_configuration: dict
        :param handler_name: The name for the handler to use
        :type handler_name: str
        :param output_path_prefix: The path of directories the configured log will be stored to. Note: in case the log
        configuration already specify a directory, it will be preserved (i.e. output_path_prefix + configuration)
        :type output_path_prefix: str
    """
    assert isinstance(log_configuration, dict), "Wrong input parameter type for 'log_configuration', expected dict"
    assert isinstance(handler_name, str), "Wrong input parameter type for 'handler_name', expected str"
    assert 'handlers' in log_configuration, "Missing handlers in the given configuration"
    assert handler_name in log_configuration['handlers'], "The requested handler is not in the configuration"
    if output_path_prefix is not None:
        assert isinstance(output_path_prefix, str), "Wrong input parameter type for 'output_path_prefix', expected str"
        assert os.path.exists(output_path_prefix), "The 'output_path_prefix' doesn't exist"
        assert os.path.isdir(output_path_prefix), "The 'output_path_prefix' is not a folder"

    # Setting the override (supposing the configuration is right)
    time_format = "%Y-%m-%d_%H-%M-%S.%f"
    time = datetime.datetime.now().strftime(time_format)
    log_file_path = log_configuration['handlers'][handler_name]['filename'].format(time)

    # Setting the folder if any
    if output_path_prefix is not None:
        log_file_path = os.path.join(output_path_prefix, log_file_path)

    # Creating missing folders from the prefix to the log
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    # Overriding configuration
    log_configuration['handlers'][handler_name]['filename'] = log_file_path
