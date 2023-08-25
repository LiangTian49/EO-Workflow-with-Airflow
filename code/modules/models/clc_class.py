class ClcClass(object):
    """
        Represents a Corine Land Cover (CLC) class
    """
    class_value: int
    class_name: str
    red: int
    green: int
    blue: int
    alpha: int

    def __init__(self, class_value, class_name, red=0, green=0, blue=0, alpha=255):
        """
            Initialize the class, the color values are expected to be among 0 and 255.
            :param class_value: The integer value representing the class which is composed by the given
            'original_classes'
            :type class_value: int
            :param class_name: The name describing the class
            :type class_name: str
            :param red: The red value
            :type red: int
            :param green: The green value
            :type green: int
            :param blue: The blue value
            :type blue: int
            :param alpha: The alpha value (opacity), the higher the more opaque
            :type alpha: int
        """
        assert isinstance(class_value, int), \
            "Wrong input parameter type for 'class_value', expected int"

        assert isinstance(class_name, str), \
            "Wrong input parameter type for 'class_name', expected str"

        self.class_value = class_value
        self.class_name = class_name
        self.set_colors(red, green, blue, alpha)

    def set_colors(self, red, green, blue, alpha):
        """
            Set the colors for representing the class.
            Each value is expected to be among 0 and 255.
            :param red: The red value
            :type red: int
            :param green: The green value
            :type green: int
            :param blue: The blue value
            :type blue: int
            :param alpha: The alpha value (opacity), the higher the more opaque
            :type alpha: int
        """
        assert 0 <= red <= 255, "The value for red is not feasible, expected among 0 and 255"
        assert 0 <= green <= 255, "The value for green is not feasible, expected among 0 and 255"
        assert 0 <= blue <= 255, "The value for blue is not feasible, expected among 0 and 255"
        assert 0 <= alpha <= 255, "The value for alpha is not feasible, expected among 0 and 255"

        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def has_colors_set(self):
        """
            Check the colors are already set, with respect the default ones (0 for colors and 1 for alpha).
            :return: True in case the colors have different values from default ones.
            :rtype: bool
        """
        return self.red != 0 or self.green != 0 or self.blue != 0 or self.alpha != 255
