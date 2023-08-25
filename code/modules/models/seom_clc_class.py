from typing import List

from modules.models.clc_class import ClcClass


class SeomClcClass(ClcClass):
    """
        Represents a Corine Land Cover (CLC) class for the SEOM project
    """
    original_classes: List[ClcClass]
    number_samples_clusters: int
    number_samples_clusters_to_keep: int

    def __init__(self, original_classes, class_value, class_name, number_samples_clusters,
                 number_samples_clusters_to_keep, red=0, green=0, blue=0, alpha=255):
        """
            Initialize the class, the color values are expected to be among 0 and 255.
            :param original_classes: The original classes which compose the current class
            :type original_classes: list
            :param class_value: The integer value representing the class which is composed by the given
            'original_classes'
            :type class_value: int
            :param class_name: The name describing the class
            :type class_name: str
            :param number_samples_clusters:
            :type number_samples_clusters: int
            :param number_samples_clusters_to_keep:
            :type number_samples_clusters_to_keep: int
            :param red: The red value
            :type red: int
            :param green: The green value
            :type green: int
            :param blue: The blue value
            :type blue: int
            :param alpha: The alpha value (opacity), the higher the more opaque
            :type alpha: int
        """
        assert isinstance(original_classes, list), \
            "Wrong input parameter type for 'original_classes', expected list"

        assert all(isinstance(c, ClcClass) for c in original_classes), \
            "Wrong input parameter type for 'original_classes', expected dict of ClcClass"

        assert isinstance(number_samples_clusters, int), \
            "Wrong input parameter type for 'number_samples_clusters', expected int"

        assert isinstance(number_samples_clusters_to_keep, int), \
            "Wrong input parameter type for 'number_samples_clusters_to_keep', expected int"

        super().__init__(class_value=class_value, class_name=class_name, red=red, green=green, blue=blue, alpha=alpha)
        self.original_classes = original_classes
        self.number_samples_clusters = number_samples_clusters
        self.number_samples_clusters_to_keep = number_samples_clusters_to_keep
