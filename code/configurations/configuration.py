"""
    The file contains a set of configurations used application-wide
"""
import numpy
import datetime
try:
    import gdal
except ImportError:
    #in Stage2022
    from osgeo import gdal
from modules.models.clc_class import ClcClass
from modules.models.seom_clc_class import SeomClcClass
from modules.models.seom_clc_map_legend import SeomClcMapLegend

VERSION = "0.3"
SAVE_CLUSTER_IMAGES = True
SAVE_TRIALS_IMAGES = True
SAVE_PREDICTION_PROBABILITIES = False
DEV_ENVIRONMENT = False
STEPS_TO_SKIP = [62]

# BATCH EXECUTION related (i.e. job array) for computing the single pipeline parameters
S2_GRANULES_TILE_DATE_COLUMN_POSITION = 2
S2_GRANULES_TILE_NAME_COLUMN_POSITION = 5
S2_GRANULES_TILE_DATE_REGEX = DATE_REGEX = r".*(([12]\d{3})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])).*"
SOURCE_REMOTE = False # Check also the CLASSIFICATION_SOURCE_DATA_PATH
DESTINATION_REMOTE = False # Check also the CLASSIFICATION_REMOTE_DATA_OUTPUT_PATH
DELETE_LOCAL_RESULTS = True
OVERWRITE_LOCAL = False
REMOTE_HOST = "192.168.163.164"
REMOTE_USERNAME = "rslab"
REMOTE_PASSWORD = ""
DATA_S2_FOLDER_NAME = "s2_images"
DATA_LC_FOLDER_NAME = "lc_product"
DATA_CMASK_FOLDER_NAME = "c_masks"
DATA_OUTPUT_FOLDER_NAME = "output"
TILE_FOLDER_PREFIX = "2018_"
CLOUD_MASK_NAME_PREFIX = "Cmask_"
CLOUD_CMASK_NAME_SUFFIX = ".tif"
LAND_COVER_NAME_SUFFIX = "_CLC_original.tif"

# IMAGES related
IMAGES_FEASIBLE_EXTENSIONS = ".tif", ".tiff"
IMAGES_S2_FILE_NAME_DATE_REGEX = r"MSI.{3}_(([12]\d{3})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01]))T\d{6}_N\d{4}_R\d{3}_T\d{2}\w{3}\.tif"
IMAGES_S2_CLOUD_MASK_FILE_NAME_DATE_REGEX = r"Cmask_\d{8}\.tif"
IMAGES_S2_NAN_VALUE = 0
IMAGES_S2_NUMPY_TYPE = numpy.uint16
IMAGES_S2_START_VALIDITY = datetime.datetime(1, 1, 1)
IMAGES_S2_END_VALIDITY = datetime.datetime(1, 12, 31)
IMAGES_CLC_PREPROCESSING_OUTPUT_FILE_NAME = "clc_converted.tif"
IMAGES_CLC_NUMPY_TYPE = numpy.uint16
IMAGES_CLC_EROSION_RADIUS = 3
IMAGES_PREDICTED_NUMPY_TYPE = numpy.uint8
IMAGES_NORMALIZED_NUMPY_TYPE = numpy.float64 # For utmost results use float64
IMAGES_OUTPUT_GDAL_TYPE = gdal.GDT_UInt16 #gdal.GDT_Byte
IMAGES_PROBABILITIES_GDAL_TYPE = gdal.GDT_Float64
IMAGES_PIXEL_RESOLUTION = 10
SEOM_COLOR_MAP_PATH = "/p/project/sdlrs/tian1/multiyearlcmaps/code/seom-python/modules/new_color_map_12.txt"

# CLOUD REMOVAL related
#CLOUD_REMOVAL_SOURCE_DATA_PATH = "/mnt/hgfs/project/data"#"/mnt/hgfs/shared/seom-cloud-mask/data"
#CLOUD_REMOVAL_REMOTE_DATA_OUTPUT_PATH = "/mnt/hgfs/shared/seom-cloud-removal/output"#"/mnt/esterno/cloud-removed"
CLOUD_REMOVAL_FEATURE_SET = [2, 1, 0, 6]
CLOUD_REMOVAL_IMAGES_TO_CONSIDER = 4
CLOUD_REMOVAL_PATCH_SIZE = 2196#2196#1098#549
CLOUD_REMOVAL_TIME_SERIES_PATCH_OVERRIDE_COLUMN_START = 0#6588#None
CLOUD_REMOVAL_TIME_SERIES_PATCH_OVERRIDE_ROW_START = 0#8784#None10000
CLOUD_REMOVAL_EROSION_SMALL_OBJECT_SIZE = 10
CLOUD_REMOVAL_KDTREE_NEIGHBOURS = 20
CLOUD_REMOVAL_DILATE_RADIUS = 3
CLOUD_REMOVAL_HIGH_QUANTILE = 0.999
CLOUD_REMOVAL_LOW_QUANTILE = 0.001

# SAMPLING related
SAMPLES_KMEANS_RANDOM_STATE = 1
SAMPLES_KMEANS_MAX_ITERATIONS = 5000
SAMPLES_KMEANS_PARALLEL_THREADS_REQUEST = 10
SAMPLES_QUANTILE_THRESHOLD = 0.75
SAMPLES_OUTPUT_FILE_NAME = "selectedPixelsClass{}.csv"
SAMPLES_OUTPUT_3Y_FILE_NAME = "selectedPixelsClass{}.csv"
SAMPLES_PRIOR_QUANTILE = 0.65
SAMPLES_PRIOR_MIN_OCCURRENCES_FOR_CLASS = 100
SAMPLES_PRIOR_MULTIPLIER = 1000
SAMPLES_PRIOR_HIGH_MODIFIER = 1000
SAMPLES_PRIOR_LOW_MODIFIER = 500
SAMPLES_STRATIFIED_TRIALS_OFFSET = 2
SAMPLES_STRATIFIED_TRIALS = 3
SAMPLES_CLUSTER_EROSION_TYPE = "SMALL_OBJECTS" # BINARY_EROSION | SMALL_OBJECTS
SAMPLES_CLUSTER_EROSION_RADIUS = 1
SAMPLES_CLUSTER_EROSION_SMALL_OBJECTS_SIZE = 2
TRIALS_OUTPUT_FILE_NAME = "trainingSet{}.csv"

# CLASSIFICATION related
# The number of features (~ bands) to use from the training set (e.g. training set on 6 images for the time-series, each
# having 10 bands, will have at most 60 bands)
#CLASSIFICATION_SOURCE_DATA_PATH = "/p/project/joaiml/gasparella1/upload01"
#CLASSIFICATION_REMOTE_DATA_OUTPUT_PATH = "/p/project/joaiml/gasparella1/classification_01"
SVM_PARALLEL_THREADS_REQUEST = 25 #25
# Static feature set using 6 images
SVM_FEATURE_SET = numpy.array([13, 1, 21, 19, 40, 39, 20, 33, 11, 9, 6, 15, 16, 36, 2, 31, 29, 30, 10, 3, 4, 26, 34, 14, 17]) - 1
SVM_C_SPACE = numpy.linspace(100, 1000, 5)
SVM_GAMMA_SPACE = numpy.linspace(0.0001, 2, 5)
SVM_COMPUTE_PROBABILITY = False
SVM_CROSS_VALIDATION_STRATEGY = 3
SVM_CLASSIFICATION_TIME_SERIES_PATCH_SIZE = 2196#549#1098#2196#5490
SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_COLUMN_START = None
SVM_CLASSIFICATION_TIME_SERIES_PATCH_OVERRIDE_ROW_START = None
SVM_THREAD_CHECK_SLEEP = 60 # seconds
CLASSIFICATION_OUTPUT_FILE_NAME = "mapPredicted_SVM_trial_{}.tif"
CLASSIFICATION_OUTPUT_FILE_NAME_LSTM = "mapPredicted_LSTM_year_{}.tif"
CLASSIFICATION_OUTPUT_FILE_NAME_LSTM_SMALLPATCH = "SmallmapPredicted_LSTM_year_{}.tif"
CLASSIFICATION_OUTPUT_FILE_NAME_RF = "mapPredicted_RF_year_{}.tif"
CLASSIFICATION_OUTPUT_FILE_NAME_RF_SMALLPATCH = "SmallmapPredicted_RF_year_{}.tif"


CLASSIFICATION_OUTPUT_FILE_NAME_TRANSFORMER = "mapPredicted_Transformer_year_{}.tif"
CLASSIFICATION_OUTPUT_FILE_NAME_TRANSFORMER_SMALLPATCH = "SmallmapPredicted_Transformer_year_{}.tif"
TOPK_OUTPUT_FILE_NAME_TRANSFORMER_SMALLPATCH = "TopK_Transformer_year_{}.tif"

#CLASSIFICATION_OUTPUT_FILE_NAME = "map2019Predictedwith2018_trial_{}.tif"
CLASSIFICATION_PROBABILITIES_OUTPUT_FILE_NAME = "mapPredicted_SVM_probabilities_trial_{}.tif"
CLASSIFICATION_OUTPUT_FINAL_FILE_NAME = "{}_seom_classified.tif"
CLASSIFICATION_OVERLAP_MAJORITY_FILE_NAME = "{}_seom_classified_overlaps_majority.tif"
CLASSIFICATION_RF_OUTPUT_FILE_NAME = "mapPredicted_RF_{}.tif"

# VALIDATION related
SAVE_PRE_PROCESSED_LUCASDB_TILE = False
#VALIDATION_SOURCE_DATA_PATH = "/mnt/hgfs/shared/seom-batch/output"#"/mnt/esterno/classification"
#VALIDATION_REMOTE_DATA_OUTPUT_PATH = "/mnt/esterno/validation"
#LUCASDB_TILES_PATH = "/mnt/hgfs/project/data/2018_Lucas_Database_Europe/lucasDatabase"#"/mnt/datadisk/s2"
#LUCASDB_TILE_SUFFIX = "_lucas_validation.tif"
LUCASDB_SEARCH_GRID_SIZE = 3    # In case of fair number the grid search will be bigger (e.g. 4 -> 5x5)
VALIDATION_SHAPEFILE_NAME = "validation.shp"
VALIDATION_CLASSIFIED_SEARCH_GRID_NAME = "classified-search-grid.tif"
VALIDATION_CSV_NAME = "validation.csv"
VALIDATION_THREAD_CHECK_SLEEP = 5 # seconds

# POST-PROCESSING related
POST_CLASSIFICATION_PARALLEL_THREADS_REQUEST = 16
POST_CLASSIFICATION_TILE_NAMES_TO_SKIP = []
KDTREE_MODEL_SAMPLES_NUMBER = 15000
POST_PROCESSING_FINAL_IMAGE_NAME = "{}_seom_classified_postprocessing.tif"
POST_PROCESS_DEBUG_CLOUDY_PIXELS_RECONSTRUCTION = False
POST_PROCESSING_MINIMUM_CLOUD_COUNT_MODEL = 0
TILES_MERGING_TARGET_PROJECTION = 32632
TILES_MERGING_SECONDARY_PROJECTION = 32633
HAUSDORFF_DISTANCE_INCLUSIVE_THRESHOLD = 0.001
ALMOST_EQUAL_DECIMAL_PRECISION = 0
SIMPLIFY_TOLERANCE = 0.1
MINIMAL_INTERSECTION_AREA_TO_KEEP = 100
THINNESS_RATIO_THRESHOLD = 0.25
BUFFER_ON_SUBTRACTING_GEOMETRY = 0.001
INTERSECTIONS_SHAPEFILE_NAME = "intersections.shp"

# LOG related
LOGGING_CONFIGURATION_FILE_PATH = "/p/project/sdlrs/tian1/multiyearlcmaps/code/seom-python/configurations/logging-classification.yaml"
LOGGING_DISPATCHER_CONFIGURATION_FILE_PATH = "/p/project/sdlrs/tian1/multiyearlcmaps/code/seom-python/configurations/logging-dispatcher.yaml"
LOGGING_CLASSIFICATION_CONFIGURATION_FILE_PATH = "/p/project/sdlrs/tian1/multiyearlcmaps/code/seom-python/configurations/logging-classification.yaml"
LOGGING_VALIDATION_CONFIGURATION_FILE_PATH = "/p/project/sdlrs/tian1/multiyearlcmaps/code/seom-python/configurations/logging-validation.yaml"
LOGGING_DATA_ANALYSIS_CONFIGURATION_FILE_PATH = "/p/project/sdlrs/tian1/multiyearlcmaps/code/seom-python/configurations/logging-data-analysis.yaml"
LOGGING_MERGE_TILES_CONFIGURATION_FILE_PATH = "//p/project/sdlrs/tian1/multiyearlcmaps/code/seom-python/configurations/logging-merge-tiles.yaml"
LOGGER_NAME = "seomLogger"
CLASSIFICATION_LOGGER_NAME = "seomLogger"
DISPATCHER_LOGGER_NAME = "seomDispatcherLogger"
CLOUD_REMOVAL_LOGGER_NAME = "cloudRemovalLogger"
VALIDATION_LOGGER_NAME = "seomValidationLogger"
DATA_ANALYSIS_LOGGER_NAME = "seomDataAnalysisLogger"
MERGING_TILES_LOGGER_NAME = "seomMergeTilesLogger"

# Defining the Corine legend (from
# https://inspire.ec.europa.eu/forum/discussion/view/5422/the-corine-land-cover-code-list-corinevalue-maintained-by-eea)
'''
clc_0 = ClcClass(0, "Unknown")
clc_111 = ClcClass(111, "Continuous urban fabric")
clc_112 = ClcClass(112, "Discontinuous urban fabric")
clc_121 = ClcClass(121, "Industrial or commercial units")
clc_122 = ClcClass(122, "Road and rail networks and associated land")
clc_123 = ClcClass(123, "Port areas")
clc_124 = ClcClass(124, "Airports")
clc_131 = ClcClass(131, "Mineral extraction sites")
clc_132 = ClcClass(132, "Dump sites")
clc_133 = ClcClass(133, "Construction sites")
clc_141 = ClcClass(141, "Green urban areas")
clc_142 = ClcClass(142, "Sport and leisure facilities")
clc_211 = ClcClass(211, "Non-irrigated arable land")
clc_212 = ClcClass(212, "Permanently irrigated land")
clc_213 = ClcClass(213, "Rice fields")
clc_221 = ClcClass(221, "Vineyards")
clc_222 = ClcClass(222, "Fruit trees and berry plantations")
clc_223 = ClcClass(223, "Olive groves")
clc_231 = ClcClass(231, "Pastures")
clc_241 = ClcClass(241, "Annual crops associated with permanent crops")
clc_242 = ClcClass(242, "Complex cultivation patterns")
clc_243 = ClcClass(243, "Land principally occupied by agriculture, with significant areas of natural vegetation")
clc_244 = ClcClass(244, "Agro-forestry areas")
clc_311 = ClcClass(311, "Broad-leaved forest")
clc_312 = ClcClass(312, "Coniferous forest")
clc_313 = ClcClass(313, "Mixed forest")
clc_321 = ClcClass(321, "Natural grasslands")
clc_322 = ClcClass(322, "Moors and heathland")
clc_323 = ClcClass(323, "Sclerophyllous vegetation")
clc_324 = ClcClass(324, "Transitional woodland-shrub")
clc_331 = ClcClass(331, "Beaches, dunes, sands")
clc_332 = ClcClass(332, "Bare rocks")
clc_333 = ClcClass(333, "Sparsely vegetated areas")
clc_334 = ClcClass(334, "Burnt areas")
clc_335 = ClcClass(335, "Glaciers and perpetual snow")
clc_411 = ClcClass(411, "Inland marshes")
clc_412 = ClcClass(412, "Peat bogs")
clc_421 = ClcClass(421, "Salt marshes")
clc_422 = ClcClass(422, "Salines")
clc_423 = ClcClass(423, "Intertidal flats")
clc_511 = ClcClass(511, "Water courses")
clc_512 = ClcClass(512, "Water bodies")
clc_521 = ClcClass(521, "Coastal lagoons")
clc_522 = ClcClass(522, "Estuaries")
clc_523 = ClcClass(523, "Sea and ocean")
'''

clc_0 = ClcClass(0, "Unknown")
clc_111 = ClcClass(1, "Continuous urban fabric")
clc_112 = ClcClass(2, "Discontinuous urban fabric")
clc_121 = ClcClass(3, "Industrial or commercial units")
clc_122 = ClcClass(4, "Road and rail networks and associated land")
clc_123 = ClcClass(5, "Port areas")
clc_124 = ClcClass(6, "Airports")
clc_131 = ClcClass(7, "Mineral extraction sites")
clc_132 = ClcClass(8, "Dump sites")
clc_133 = ClcClass(9, "Construction sites")
clc_141 = ClcClass(10, "Green urban areas")
clc_142 = ClcClass(11, "Sport and leisure facilities")
clc_211 = ClcClass(12, "Non-irrigated arable land")
clc_212 = ClcClass(13, "Permanently irrigated land")
clc_213 = ClcClass(14, "Rice fields")
clc_221 = ClcClass(15, "Vineyards")
clc_222 = ClcClass(16, "Fruit trees and berry plantations")
clc_223 = ClcClass(17, "Olive groves")
clc_231 = ClcClass(18, "Pastures")
clc_241 = ClcClass(19, "Annual crops associated with permanent crops")
clc_242 = ClcClass(20, "Complex cultivation patterns")
clc_243 = ClcClass(21, "Land principally occupied by agriculture, with significant areas of natural vegetation")
clc_244 = ClcClass(22, "Agro-forestry areas")
clc_311 = ClcClass(23, "Broad-leaved forest")
clc_312 = ClcClass(24, "Coniferous forest")
clc_313 = ClcClass(25, "Mixed forest")
clc_321 = ClcClass(26, "Natural grasslands")
clc_322 = ClcClass(27, "Moors and heathland")
clc_323 = ClcClass(28, "Sclerophyllous vegetation")
clc_324 = ClcClass(29, "Transitional woodland-shrub")
clc_331 = ClcClass(30, "Beaches, dunes, sands")
clc_332 = ClcClass(31, "Bare rocks")
clc_333 = ClcClass(32, "Sparsely vegetated areas")
clc_334 = ClcClass(33, "Burnt areas")
clc_335 = ClcClass(34, "Glaciers and perpetual snow")
clc_411 = ClcClass(35, "Inland marshes")
clc_412 = ClcClass(36, "Peat bogs")
clc_421 = ClcClass(37, "Salt marshes")
clc_422 = ClcClass(38, "Salines")
clc_423 = ClcClass(39, "Intertidal flats")
clc_511 = ClcClass(40, "Water courses")
clc_512 = ClcClass(41, "Water bodies")
clc_521 = ClcClass(42, "Coastal lagoons")
clc_522 = ClcClass(43, "Estuaries")
clc_523 = ClcClass(44, "Sea and ocean")

clc_map_legend = [
    clc_0
    , clc_111
    , clc_112
    , clc_121
    , clc_122
    , clc_123
    , clc_124
    , clc_131
    , clc_132
    , clc_133
    , clc_141
    , clc_142
    , clc_211
    , clc_212
    , clc_213
    , clc_221
    , clc_222
    , clc_223
    , clc_231
    , clc_241
    , clc_242
    , clc_243
    , clc_244
    , clc_311
    , clc_312
    , clc_313
    , clc_321
    , clc_322
    , clc_323
    , clc_324
    , clc_331
    , clc_332
    , clc_333
    , clc_334
    , clc_335
    , clc_411
    , clc_412
    , clc_421
    , clc_422
    , clc_423
    , clc_511
    , clc_512
    , clc_521
    , clc_522
    , clc_523
]

# Defining the SEOM legend (set the colors to make a specific legend for the class - it will not be changed)

# 12 classes
seomClcLegend = SeomClcMapLegend()
seomClcLegend.add_element(SeomClcClass([clc_0], 0, "Unknown", 0, 0))
seomClcLegend.add_element(SeomClcClass([clc_111, clc_112, clc_121, clc_122], 1, "Urban", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_231, clc_321, clc_333], 2, "Grass", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_211, clc_212, clc_221, clc_222, clc_223, clc_241, clc_242], 3, "Crops", 4, 4))
seomClcLegend.add_element(SeomClcClass([clc_213], 4, "Rice", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_131, clc_332, clc_331], 5, "Mineral, Rocks and Sand", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_311, clc_323], 6, "Broadleaves", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_312], 7, "Conifers", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_322], 8, "Shrub", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_521], 9, "Lagoons", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_512], 10, "Lake", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_335], 11, "Snow", 4, 3))
'''

# 16 classes
seomClcLegend = SeomClcMapLegend()
seomClcLegend.add_element(SeomClcClass([clc_0], 0, "Unknown", 0, 0))
seomClcLegend.add_element(SeomClcClass([clc_111, clc_112, clc_121, clc_122], 1, "Urban", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_231, clc_321, clc_333], 2, "Grass", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_211], 3, "Not irrigated crops", 4, 4))
seomClcLegend.add_element(SeomClcClass([clc_212], 4, "Irrigated crops", 4, 4))
seomClcLegend.add_element(SeomClcClass([clc_221, clc_222, clc_223, clc_241, clc_242], 5, "Annual crops", 4, 4))
seomClcLegend.add_element(SeomClcClass([clc_213], 6, "Rice", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_131], 7, "Mineral", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_332], 8, "Rocks", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_331], 9, "Sand", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_311, clc_323], 10, "Broadleaves", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_312], 11, "Conifers", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_322], 12, "Shrub", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_521], 13, "Lagoons", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_512], 14, "Lake", 4, 3))
seomClcLegend.add_element(SeomClcClass([clc_335], 15, "Snow", 4, 3)) 
'''