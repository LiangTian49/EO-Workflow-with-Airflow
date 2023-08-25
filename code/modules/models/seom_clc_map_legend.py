from typing import List

from modules.exceptions.SeomException import SeomException
from modules.models.seom_clc_class import SeomClcClass


class SeomClcMapLegend(object):
    """
        Represents the Corine Land Cover Map Legend for SEOM (that is the contained representation for the .
    """
    seom_classes: List[SeomClcClass]

    def __init__(self):
        self.seom_classes = []

    def add_element(self, seom_clc_class):
        """
            Adding a new class to the legend
            :param seom_clc_class: The class to add
            :type seom_clc_class: SeomClcClass
        """
        isinstance(seom_clc_class, SeomClcClass), \
           "Wrong input parameter type for 'seom_clc_class', expected SeomClcClass"

        if self.is_class_code_in_use(seom_clc_class.class_value):
            raise SeomException("Class code " + str(seom_clc_class.class_value) + " is being used by multiple classes")

        self.seom_classes.append(seom_clc_class)

    def is_class_code_in_use(self, seom_class_code):
        """
            Given the class code for a SEOM class, it will return true in case there is already a class in the legend
            having the same value.
            :param seom_class_code: The SEOM class code to check
            :type seom_class_code: int
            :return: True in case the class code is already in use
            :rtype: str
        """
        assert isinstance(seom_class_code, int), "Wrong type for 'seom_class_code', expected int"

        classes = [x for x in self.seom_classes if x.class_value == seom_class_code]
        return len(classes) > 0

    def get_colors_map(self):
        """
            Retrieve the legend color map.
            :return: A map indexed by the code for a tuple of colors in the format (red, green, blue, alpha) with values
            among 0 and 255 included.
            :rtype: dict
        """
        result = dict()
        for seom_class in self.seom_classes:
            result[seom_class.class_value] = (seom_class.red, seom_class.green, seom_class.blue, seom_class.alpha)

        return result

    def get_seom_class_codes(self):
        """
            Retrieve the class codes used in the legend, except the '0' which is the "Unknown".
            :return: The SEOM class codes
            :rtype: list of int
        """
        seom_class_codes = [x.class_value for x in self.seom_classes if x.class_value > 0]
        return seom_class_codes

    def get_max_class_code(self):
        """
            Retrieve the maximum value for the SEOM class code in the legend
            :return: The maximum value of class code in the legend
            :rtype: int
        """
        seom_class_codes = self.get_seom_class_codes()
        max_class_code = max(seom_class_codes)
        return max_class_code

    def get_seom_class_name_by_code(self, seom_class_code):
        """
            Given the class code for a SEOM class, it will return its associated label/name.
            :param seom_class_code: The SEOM class code for which retrieve the label/name
            :type seom_class_code: int
            :return: The SEOM class label/code for the given code
            :rtype: str
        """
        assert isinstance(seom_class_code, int), "Wrong type for 'seom_class_code', expected int"

        class_names = [x.class_name for x in self.seom_classes if x.class_value == seom_class_code]
        if len(class_names) > 1:
            raise SeomException("Class code " + str(seom_class_code) + " is being used by multiple classes")

        return class_names[0]
