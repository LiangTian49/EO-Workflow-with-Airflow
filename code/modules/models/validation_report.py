from configurations import configuration
from modules.exceptions.SeomException import SeomException


class ValidationReport(object):
    """
        Class for managing the validation of the classified map.
    """
    # The classes to report
    classes: dict

    # The indexes for the tuple in the map values
    class_code_value_index = 0
    class_label_index = 1
    samples_count_index = 2
    classified_under_samples_count_index = 3
    tile_classified_count_index = 4
    high_occurrences_index = 5
    medium_occurrences_index = 6
    low_occurrences_index = 7

    # The thresholds for counting the elements
    high_occurrences_threshold = 0.75
    medium_occurrences_threshold = 0.5
    low_occurrences_threshold = 0.25

    # Headers
    class_code_header = "SEOM class code"
    class_label_header = "SEOM class name"
    samples_count_header = "LucasDB samples for tile"
    classified_under_samples_count_header = "Elements for the class in the classified map beneath the samples (by using search grid)"
    tile_classified_count_header = "Elements in the classified map for the class"
    high_occurrences_header = "Samples having >= " + str(high_occurrences_threshold * 100) + "% classified pixels beneath"
    medium_occurrences_header = "Samples having >= " + str(medium_occurrences_threshold * 100) + "% classified pixels beneath"
    low_occurrences_header = "Samples having >= " + str(low_occurrences_threshold * 100) + "% classified pixels beneath"

    def __init__(self):
        self.classes = dict()

    def add_elements_sample_count(self, class_code_value, elements_count=1):
        """
            Add the count of elements for the class from the sample
            :param class_code_value: The SEOM class code value to report
            :type class_code_value: int
            :param elements_count: The number of elements to add to the total count for the class
            :type elements_count: int
        """
        assert isinstance(class_code_value, int), "Wrong type for 'class_code_value', expected int"
        assert class_code_value in self.classes, "The 'class_code_value' has not been initialized yet"
        assert isinstance(elements_count, int), "Wrong type for 'elements_count', expected int"

        self.classes[class_code_value][self.samples_count_index] += elements_count

    def add_elements_sample_classified_count(self, class_code_value, elements_count):
        """
            Add the count of elements for the class from the classification beneath the sample, considering the search
            grid (e.g. 3x3)
            :param class_code_value: The SEOM class code value to report
            :type class_code_value: int
            :param elements_count: The number of elements to add to the total count for the class
            :type elements_count: int
        """
        assert isinstance(class_code_value, int), "Wrong type for 'class_code_value', expected int"
        assert class_code_value in self.classes, "The 'class_code_value' has not been initialized yet"
        assert isinstance(elements_count, int), "Wrong type for 'elements_count', expected int"

        self.classes[class_code_value][self.classified_under_samples_count_index] += elements_count

    def add_elements_tile_count(self, class_code_value, elements_count):
        """
            Add the count of elements for the class in the classified tile (whole)
            :param class_code_value: The SEOM class code value to report
            :type class_code_value: int
            :param elements_count: The number of elements to add to the total count for the class
            :type elements_count: int
        """
        assert isinstance(class_code_value, int), "Wrong type for 'class_code_value', expected int"
        assert class_code_value in self.classes, "The 'class_code_value' has not been initialized yet"
        assert isinstance(elements_count, int), "Wrong type for 'elements_count', expected int"

        self.classes[class_code_value][self.tile_classified_count_index] += elements_count

    def get_csv_header(self, delimiter=","):
        """
            Compose a string to be a csv header
            :param delimiter: The delimiter character to use
            :type delimiter: str
            :return: The string representing the header, each header column separated by the given delimiter
            :rtype: str
        """
        result = self.class_code_header + delimiter + self.class_label_header + delimiter + self.samples_count_header +\
                 delimiter + self.classified_under_samples_count_header + delimiter + \
                 self.tile_classified_count_header + delimiter + self.high_occurrences_header + delimiter + \
                 self.medium_occurrences_header + delimiter + self.low_occurrences_header

        return result

    def get_data(self):
        """
            Retrieve the acquired data.
            :return: Ordered list of tuples
            :rtype: list of tuple
        """
        result = [(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]) for x in self.classes.values()]
        result = sorted(result, key=lambda element: element[0])

        return result

    def initialize_class(self, class_code_value):
        """
            Initialize in the report the entry for the given class code
            :param class_code_value: The SEOM class code value to report
            :type class_code_value: int
        """
        assert isinstance(class_code_value, int), "Wrong type for 'class_code_value', expected int"
        assert not class_code_value in self.classes, "The 'class_code_value' has already been initialized"

        self.classes[class_code_value] = dict()
        self.classes[class_code_value][self.class_code_value_index] = class_code_value
        class_label = configuration.seomClcLegend.get_seom_class_name_by_code(class_code_value)
        self.classes[class_code_value][self.class_label_index] = class_label
        self.classes[class_code_value][self.samples_count_index] = 0
        self.classes[class_code_value][self.classified_under_samples_count_index] = 0
        self.classes[class_code_value][self.tile_classified_count_index] = 0
        self.classes[class_code_value][self.high_occurrences_index] = 0
        self.classes[class_code_value][self.medium_occurrences_index] = 0
        self.classes[class_code_value][self.low_occurrences_index] = 0

    def is_class_existing(self, class_code_value):
        """
            Check if the class code is already being initialized
            :param class_code_value: The SEOM class code value to report
            :type class_code_value: int
        """
        assert isinstance(class_code_value, int), "Wrong type for 'class_code_value', expected int"

        return class_code_value in self.classes

    def validate_search_grid_element(self, sample_class_code, classification_statistics):
        """
            Acquire the statistics for a given sample
            :param sample_class_code: The SEOM class code value the sample is about
            :type sample_class_code: int
            :param classification_statistics: List with statistics for the classification beneath the sample.
            Each element will have: (class_code, count, percentage).
            The list will be ordered from the more abundant class_code to the lesser.
            :type classification_statistics: list of (int, int, float)
        """
        assert isinstance(sample_class_code, int), "Wrong type for 'class_code_value', expected int"

        # Adding the sample being observed
        self.add_elements_sample_count(sample_class_code)

        # Getting the elements count in the classification statistics for the sample class code (observed)
        classified_elements_same_class_code_candidates = [x for x in classification_statistics if x[0] == sample_class_code]
        if len(classified_elements_same_class_code_candidates) > 1:
            raise SeomException("There are multiple value in the statistics for the same class code")

        # In case there aren't any elements in the classification statistics, then do not update anything else
        if len(classified_elements_same_class_code_candidates) == 0:
            return

        # The actual classification statistic for the sample class code
        class_code_statistic = classified_elements_same_class_code_candidates[0]

        # Adding the elements in the statistics for the class
        self.add_elements_sample_classified_count(sample_class_code, class_code_statistic[1])

        # Updating based on class code frequency in the statistics (#elements found in the search grid)
        class_code_frequency = class_code_statistic[2]

        # Update if greater or equal of the high threshold
        if class_code_frequency >= self.high_occurrences_threshold:
            self.classes[sample_class_code][self.high_occurrences_index] += 1
            return

        # Update if greater or equal of the medium threshold
        if class_code_frequency >= self.medium_occurrences_threshold:
            self.classes[sample_class_code][self.medium_occurrences_index] += 1
            return

        # Update if greater or equal of the low threshold
        if class_code_frequency >= self.low_occurrences_threshold:
            self.classes[sample_class_code][self.low_occurrences_index] += 1
            return
