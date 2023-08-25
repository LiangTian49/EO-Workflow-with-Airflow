class SpatialComputationParameters(object):
    """
        The spatial computations parameters to use during spatial processing
    """
    almost_equal_decimal_precision: float
    buffer_on_cleaning_geometry: float
    buffer_on_substracting_geometry: float
    epsg_code: int
    hausdorff_distance_inclusive_threshold: float
    minimal_intersection_area_to_keep: float
    simplify_tolerance: float
    thinness_ratio_threshold: float

    def __init__(self):
        almost_equal_decimal_precision = None
        buffer_on_cleaning_geometry = None
        buffer_on_substracting_geometry = None
        epsg_code = None
        hausdorff_distance_inclusive_threshold = None
        minimal_intersection_area_to_keep = None
        simplify_tolerance = None
        thinness_ratio_threshold = None
