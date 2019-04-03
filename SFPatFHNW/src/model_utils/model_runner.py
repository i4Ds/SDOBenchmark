# General
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
from datetime import datetime
from collections import namedtuple

# Tensorflow
import tensorflow as tf

# Keras
from keras import backend as KB
from keras.models import Model
from keras.layers import Input
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import SGD

# Our
from model_utils.keras_generator import SDOBenchmarkGenerator
from evaluation.callbacks_caller import TensorBoardCallbacksCaller
from evaluation.callbacks_plots import *
from evaluation.callbacks_mes import *
from evaluation.callbacks_r2s import *


LabelFuncs = namedtuple('LabelFuncs', ['label', 'unlabel'])

LABEL_FUNCS_LOGEXP = LabelFuncs(lambda y: np.log(y), lambda y: np.exp(y))


def bin_func_sum(y):
    return np.sum(y, -1)


"""
    An important note on encoding functions that turn the prediction of the peak flux into "Regression by Classification":
Unfortunately, our models start behaving unpredictably when any encoding function is used that turns the problem into one of logistic regression. That is, if the encoding function turns one value into many values, such as with threshold encoding or one-hot classification encoding the models start behaving in the following way. After initialisation, the models make predictions far off the optimal baseline, and as expected, different predictions for different samples. However, after the first training epoch, the model still makes unusably bad predictions, but now it makes the same or almost the same prediction for all samples. After additional training epochs, the value predicted for all samples changes, but still stays the same across samples.Moreover, this prediction does not seem to converge to the optimal static prediction, the baseline.
"""

def get_label_funcs_threshold(bin_count=10, lower=1e-9, upper=1e-3, bin_func=bin_func_sum):
    """
    :param bin_count: Number of different thresholds to encode the goes flux into.
    :param lower: Lower limit for valid goes flux values.
    :param upper: Upper limit for valid goes flux values.
    :param bin_func: Function used to return the list of outputs the model makes to a single bin, which will then be
                     transformed back to a goes flux value.
    """
    bin_labels = np.zeros((bin_count, bin_count - 1))
    for i in range(bin_count):
        bin_labels[i][:i] = 1

    low = np.log10(lower)
    up = np.log10(upper)

    def label_func(y):
        val = np.log10(y)
        rawbin = (val - low) / (up - low) * bin_count
        index = np.floor(rawbin).astype(int)
        return bin_labels[np.clip(index, 0, bin_count - 1)]

    def unlabel_func(y):
        val = bin_func(y)
        print(val)
        exponent = (((val / bin_count) * (up - low)) + low)
        return 10 ** exponent

    return LabelFuncs(label_func, unlabel_func)


# TODO: set these up so that they only cover the relevant range (which is e-3 to e-9 afaik)
def get_label_funcs_onehot(classes_count):
    categories = to_categorical(np.arange(classes_count))

    return  LabelFuncs(
                lambda y: categories[np.clip(np.floor(classes_count + (classes_count / 10) * np.log10(y)).astype(int),0,classes_count-1)],
                lambda x: (10. ** (np.argmax(x, axis=-1) / (classes_count / 10.))) / (10 ** 10))

class ModelRunner(object):

    ### Initialization
    def __init__(self, model_name: str, cuda_visible_devices: str = "", log_device_placement: bool = False):
        """
        :param model_name: Used to label the evaluation logs
        :param cuda_visible_devices: string, containing comma separated integers
                                     Only the GPUs listed in this argument will be used during training. Their indexing starts at 0
        :param log_device_placement: If set to true, Keras will report if it is running the model on the CPU or the GPU
        """
        self.model_name = model_name
        self._init_session(cuda_visible_devices, log_device_placement)

    def _init_session(self, cuda_visible_devices, log_device_placement):
        # Configure GPUs used
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

        # Start session
        config = tf.ConfigProto(log_device_placement = log_device_placement)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Set Session for Keras
        KB.set_session(sess)

        # Save Session
        self.session = sess

    ### Input Initialization
    main_input = None
    date_input = None
    def init_input(self,
            channels               = ['magnetogram'],
            include_date: bool     = False,
            sample_data_only: bool = False,
            only_last_slice: bool  = False,
            augment_data: bool     = True,
            label_funcs            = None):
        """
        :param channels: list of strings, any subset from: ["94", "131", "171", "193", "211", "304", "335", "1700",
                         "continuum", "magnetogram"]
                         This setting determines how many data SDO data channels are provided to the network. This has
                         therefore a significant effect on the input size and thus training time.
        :param include_date: Determines whether or not the date when the satellite images were taken should be  prepared
                             to be used by the model script as part of a layer.
        :param sample_data_only: If this is switched to true, only the very small sample dataset will be used during
                                training. This is useful during the development and debugging of a model.
        :param only_last_slice: Determines if either images from all four time slices before the prediction period or
                                only the last one should be provided to the network.
        :param augment_data: This switch en- or disables all data augmentation, such as vertical flipping.
        :param label_funcs: There are three statically available labeling function pairs: logarithmic encoding,
                            threshold encoding and onehot encoding.
                            If both a labeling function and an unlabeling function is provided, the labels are
                            transformed before training using the labeling function and predictions are transformed back
                            using the unlabeling function to provide predictions of the goes flux. If the labeling
                            function returns a list of values the “ModelRunner” uses categorical cross-entropy as loss
                            function instead of mean absolute error.
        :return: start layer to build upon
        """
        self.label_func = label_funcs.label if label_funcs is not None else None
        self.unlabel_func = label_funcs.unlabel if label_funcs is not None else None

        main_input_dim = self._create_main_input_dim(only_last_slice, len(channels))

        self._init_generators(main_input_dim, channels, include_date, sample_data_only, only_last_slice, augment_data)

        if (include_date):
            self.date_input = Input(shape=(1,), name='date_input')

        self.main_input = Input(shape=main_input_dim, name='images_input')
        return self.main_input

    def get_date_input(self):
        assert (self.main_input is not None), "Run init_input() to allow ModelRunner to initialize inputs."
        assert (self.date_input is not None), "Run init_input() with include_date = True to allow ModelRunner to initialize date input."

        return self.date_input

    def _create_main_input_dim(self,only_last_slice, channels_count):
        if (not only_last_slice):
            return (4, 256, 256, channels_count)
        else:
            return (256, 256, channels_count)

    def _init_generators(self, main_input_dim, channels, include_date, sample_data_only, only_last_slice, augment_data):

        # Determine parameters for generators
        if (not sample_data_only):
            base_data_path = '../data/full/'
        else:
            base_data_path = '../data/sample/'

        generator_params = {'dim': main_input_dim,
                            'channels': channels,
                            'data_format': 'channels_first',
                            'shuffle': True,
                            'augment': augment_data,
                            'include_date': include_date,
                            'only_last_slice': only_last_slice}

        if (self.label_func is not None):
            generator_params['label_func'] = self.label_func

        self.training_generator = SDOBenchmarkGenerator(os.path.join(base_data_path, 'train'), **(generator_params))

        generator_params['augment'] = False
        self.validation_generator = SDOBenchmarkGenerator(os.path.join(base_data_path, 'test'), **(generator_params))

    ### Training
    def train(self, layers,
            dont_train: bool        = False,
            optimizer               = SGD(),
            epochs                  = 100,
            batch_size              = 10,
            use_multiprocessing     = False,
            save_wheights           = True,
            detailed_tensorboard    = True,
            eval_to_console_after_init = True,
            eval_to_console_after_train = True):
        """
        :param layers:
        :param dont_train: Runs all evaluation and logging as usual, but switches all layers to non-trainable thus
                           enabling fairly fast full stack tests of any and all code.
        :param optimizer: Sets the optimizer used during training.
        :param epochs: Number of epochs, i.e. how many times the full training data set should be run through during training.
        :param batch_size: Size of the mini batches the training data set should be split into for parallel training
                           during each epoch. If set to one, no mini-batching is performed.
        :param use_multiprocessing: Whether or not multiple threads should be used during training.
        :param save_wheights: If changed to false, the weights for the model will not be saved to
                              “../models_trained/[model_name]_[date & time].hdf5”.
        :param detailed_tensorboard: None of the metrics run by the “TensorBoardCallbacksCaller”  will be added to the
                                     TensorBoard log if this is set to false.
        :param eval_to_console_after_init: By default, the “ModelRunner” prints a detailed evaluation to the console after initialisation.
        :param eval_to_console_after_train: By default, the “ModelRunner” prints a detailed evaluation to the console after training.
        :return: train history
        """
        assert (self.main_input is not None), "Run initialize_input() to allow ModelRunner to initialize for training!"

        self.predict_gen_params = {"use_multiprocessing": use_multiprocessing, "max_queue_size": batch_size, "workers": batch_size*2}

        self.training_generator.batch_size = batch_size
        self.validation_generator.batch_size = batch_size

        self._init_model(layers, dont_train, optimizer)
        callbacks = self._create_callbacks(save_wheights, detailed_tensorboard)

        if eval_to_console_after_init:
            self.evaluate(verbose=True, epoch='_begin')

        history = self.model.fit_generator(generator        = self.training_generator,
                                      validation_data       = self.validation_generator,
                                      steps_per_epoch       = len(self.training_generator),
                                      validation_steps      = len(self.validation_generator),
                                      epochs                = epochs,
                                      class_weight          = self.classes_weights,
                                      callbacks             = callbacks,
                                      **(self.predict_gen_params))
        
        if eval_to_console_after_train and not detailed_tensorboard:
            self.evaluate(verbose=True, epoch='_end')
        
        return history

    model = None
    def _init_model(self, layers, dont_train, optimizer):

        # Determine if regression or classification
        is_regression = self.label_func is None or np.isscalar(self.label_func(1e-4))

        if (is_regression):
            loss='mean_absolute_error'
        else:
            loss='categorical_crossentropy'
            self._init_class_wheights()

        # Construct model
        if self.date_input is None: # date_input is not used in model
            self.model = Model(inputs=[self.main_input], outputs=[layers])

        else: # date_input is used in model
            self.model = Model(inputs=[self.main_input, self.date_input], outputs=[layers])

        if dont_train:
            for layer in self.model.layers:
                layer.trainable = False

        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.summary()

    classes_weights = None
    def _init_class_wheights(self):
        category_balance = np.sum(self.label_func(self.training_generator.data['peak_flux']), axis=0)

        category_weights = category_balance
        category_weights[np.nonzero(category_balance)] = np.sum(category_balance) / (category_balance[np.nonzero(category_balance)])
        category_weights[np.isinf(category_weights)] = 0

        self.classes_weights = dict(zip(range(len(category_balance)), category_weights))

    def _create_callbacks(self, save_wheights, detailed_tensorboard):
        callbacks = []

        now_formatted = datetime.now().strftime('%d.%m.%Y_%H.%M.%S')

        # Ensure required directories exist
        if not os.path.isdir('../logs'):
            os.makedirs('../logs')
        if not os.path.isdir('../models_trained'):
            os.makedirs('../models_trained')

        log_dir = '../logs/{}/{}'.format(self.model_name, now_formatted)
        self.dir = log_dir
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True))

        if save_wheights:
            callbacks.append(ModelCheckpoint('../models_trained/{}_{}.hdf5'.format(self.model_name, now_formatted), monitor='val_loss', verbose=1, save_best_only=True))

        if detailed_tensorboard:
            if self.unlabel_func is None:
                callbacks.append(TensorBoardCallbacksCaller(callbacks[0], self.predict_gen_params, generator=self.validation_generator, verbose=True))
            else:
                callbacks.append(TensorBoardCallbacksCaller(callbacks[0], self.predict_gen_params, generator=self.validation_generator, unlabel_func=self.unlabel_func, verbose=True))

        return callbacks

    def evaluate(self, verbose=False, epoch='eval'):
        """
        :param verbose: Determines whether or not additional data should be printed to the console, such as all true
                        and predicted labels.
        :param epoch: Label for the custom plots, see “5.4 Usage, Evaluation & Weights”.
        :return:
        """
        assert (self.model is not None), "Run train() to allow ModelRunner to initialize for evaluating!"

        #prepare true labels
        y_true = self.validation_generator.get_labels()

        # prepare predicted labels
        y_pred = self.model.predict_generator(generator=self.validation_generator, **(self.predict_gen_params), verbose=True)
        if self.unlabel_func is not None:
            y_pred = self.unlabel_func(y_pred)
        y_pred = y_pred.flatten()


        if (verbose):
            print("Model Runner: Evaluate:")

            print("True Labels:")
            print(y_true)

            print("Predicted Labels:")
            print(y_pred)

        callbacks=[mae_on_epoch_end,mse_on_epoch_end,rmse_on_epoch_end,
                r2_on_epoch_end, adjustedR2FuncFactory(self.validation_generator.dim),
                residualsplot_on_epoch_end, boxplots_on_epoch_end]
        for callback in callbacks:
            callback(self, epoch, y_true, y_pred)

    # This imitates this being a callbacks_caller, do not use!
    dir = None
    def add_simple_value(self, name, value, epoch):
        print(f'{name}: {value}')
