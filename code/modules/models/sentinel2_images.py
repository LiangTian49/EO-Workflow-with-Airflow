class Sentinel2Images(object):
    """
        Represents the Sentinel 2 images to process.
        It is a descriptor, so it contains some metadata shared among all the images
    """

    def __init__(self):
        # The path containing all the images
        self.path = None
        # The map containing the Sentinel 2 file name and the matrix representing it
        # Note: the insertion order is remembered https://docs.python.org/3/library/collections.html#ordereddict-objects
        self.name_matrix_map = dict()
        # The map containing the Sentinel 2 file name and the matrix representing the associated cloud cover
        self.name_cloud_covers_matrix_map = dict()
        # The gdal image geo-transform
        self.geo_transform = None
        # The gdal image projection
        self.projection = None
        # The size of pixels in the X direction (width)
        self.x_size = None
        # The size of pixels in the Y direction (height)
        self.y_size = None
        # The number of bands
        self.bands_size = None

    def get_matrix_by_name(self, image_name):
        """
            :return: Retrieve the matrix corresponding to the given name
            :param image_name: The name of the image to retrieve as matrix
            :type image_name: str
            :rtype: numpy.ndarray
        """
        if image_name in self.name_matrix_map.keys():
            return self.name_matrix_map.get(image_name)

        return None
