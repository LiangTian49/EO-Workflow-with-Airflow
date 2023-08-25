import os

from modules.exceptions.SeomException import SeomException


class InputParameters(object):
    """
        The class represents all the input parameters given to the application
    """

    def __init__(self):
        self.is_to_save_intermediate_outputs = True
        self.s2_data_path = None
        self.s2_samples_image_name = None
        self.cloud_mask_image_path = None
        self.corine_land_cover_data_path = None
        self.__output_path = None
        self.__temporary_path = None

    @property
    def output_path(self):
        """
            :return: The path to save the output results (locally)
            :rtype: str
        """
        return self.__output_path

    @output_path.setter
    def output_path(self, value):
        """
            :param value: The path for saving the output results (locally)
            :type value: str
        """
        absolute_path = os.path.expanduser(value)
        if not os.path.exists(absolute_path) or not os.path.isdir(absolute_path):
            raise SeomException("The given value for storing output results doesn't exist or it is wrong")
        self.__output_path = absolute_path

    @property
    def temporary_path(self):
        """
            :return: The path to save temporary data (e.g. downloading from remote before loading into memory) (locally)
            :rtype: str
        """
        return self.__temporary_path

    @temporary_path.setter
    def temporary_path(self, value):
        """
            :param value: The path for saving the temporary data (e.g. downloading from remote before loading into
            memory) (locally)
            :type value: str
        """
        absolute_path = os.path.expanduser(value)
        if not os.path.exists(absolute_path) or not os.path.isdir(absolute_path):
            raise SeomException("The given value for storing temporary data doesn't exist or it is wrong")
        self.__temporary_path = absolute_path

    def get_s2_samples_image_path(self):
        """
            :return: The full path of the Sentinel 2 samples image
            :rtype: str
        """
        return os.path.join(self.s2_data_path, self.s2_samples_image_name)
