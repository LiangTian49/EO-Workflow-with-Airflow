import numpy
import logging
from modules.models.input_parameters import InputParameters
from modules.models.sentinel2_images import Sentinel2Images


class ApplicationContext(object):
    """
        Represents the application execution context
    """
    # The logger
    logger: logging.Logger
    # The input parameters
    input_parameters: InputParameters
    # Tile year
    tile_year = int
    # Tile name
    tile_name = str
    # Original CLC map which may be loaded by the dispatcher
    clc_original_image: numpy.ndarray
    # The cloud mask for the Sentinel2 sample image
    cloud_mask_image: numpy.ndarray
    # Result of STEP1 (preprocessing S2) or from dispatcher
    s2_images: Sentinel2Images
    # Results of STEP2 (preprocessing LC)
    clc_map_converted_matrix: numpy.ndarray
    # Result of STEP3
    selected_pixels_for_class: dict
    # Result of STEP4
    training_sets: list
    # Result of STEP5
    predicted_images: list
    # Result of post-processing (majority rules) on 'predicted_images'
    predicted_image: numpy.ndarray
    # The remote server password
    remote_server_password: str
    # The minValues override (previously computed)
    min_values: list
    # The maxValues override (previously computed)
    max_values: list
    # The c SVM parameter override
    svm_c_parameters: list
    # The gamma SVM parameter override
    svm_gamma_parameters: list
    # The numbers corresponding to the step to skip (data provided by other mean)
    steps_to_skip: list
    # The number of acquisition up to which the training set is loaded for the RF model training    
    id_acquisition = int
    # Output path
    output_folder: str
    # Name of the model
    modelname: str
    # Name of the model
    training_mode: str
    # Path to saved model
    path_model: str
    # Year to predict
    year_to_predict: int
    # List of images for multiple predictions
    list_images: list
    # use id_acquisition
    use_id_acquisition: bool
    # Hyperparameters
    batchsize: int
    hidden_dims: int
    num_layers: int
    dropout: int
    bidirectional: bool
    use_layernorm: bool
    train_model: bool
    learning_rate: int
    weight_decay: int
    epochs: int
    dd_model: int
    dn_head: int
    dn_layers: int
    dd_inner: int
    dactivation: str
    # Set if normalizing or scaling data
    range_data: str
    name_folder_experiment: str
    name_folder_subarea: str
    training_single_acquisition: str
    fixed_len_prediction_window: str
    pred_win_len: int
    consistency_factor: int

    def __init__(self):
        self.logger = None
        self.tile_year = 0
        self.tile_name = None
        self.input_parameters = InputParameters()
        self.clc_original_image = None
        self.cloud_mask_image = None
        self.s2_images = None
        self.clc_map_converted_matrix = None
        self.selected_pixels_for_class = None
        self.training_sets = None
        self.predicted_images = None
        self.predicted_image = None
        self.remote_server_password = None
        self.min_values = None
        self.max_values = None
        self.svm_c_parameters = None
        self.svm_gamma_parameters = None
        self.steps_to_skip = None
        self.list_images = None
        self.id_acquisition = 0
        self.use_id_acquisition = False
        self.output_folder = None
        self.name_folder_experiment = None
        self.name_folder_subarea = None
        self.training_single_acquisition = None
        self.fixed_len_prediction_window = None
        self.pred_win_len = 5
        self.consistency_factor = 0

        self.modelname = None
        self.training_mode = None
        self.batchsize = 512
        self.hidden_dims=128
        self.num_layers=4
        self.dropout=0.1
        self.bidirectional=True
        self.use_layernorm=True
        self.train_model=True
        self.learning_rate = 3e-4
        self.weight_decay = 1e-1
        self.epochs = 150
        self.year_to_predict = 2018
        self.range_data = None
        self.path_model = None
        self.dd_model = 0
        self.dn_head = 0
        self.dn_layers = 0
        self.dd_inner = 0
        self.dactivation = None
