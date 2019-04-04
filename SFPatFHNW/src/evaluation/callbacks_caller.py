import tensorflow as tf
import keras
import numpy as np

# Default Callbacks Imports
from evaluation.callbacks_plots import *
from evaluation.callbacks_mes import *
from evaluation.callbacks_r2s import *


class TensorBoardCallbacksCaller(keras.callbacks.Callback):
    def __init__(self, tensorBoard, predict_gen_params, validation_set=None, generator=None, unlabel_func=None, verbose=False, callbacks=[mae_on_epoch_end,mse_on_epoch_end,rmse_on_epoch_end,
            r2_on_epoch_end,
            residualsplot_on_epoch_end, boxplots_on_epoch_end]):
        """
        :param tensorBoard: TensorBoard Logging Instance
                            Used to find the logging directory and provides access to the TensorBoard Log-Writter.
        :param predict_gen_params: Dictionary of options to run the model on the evaluation data.
                                   See the “ModelRunner” source code to find out about the construction of this dictionary.
        :param validation_set: Dataset of the from [X, Y], suitable to be passed to the model.
                               Either validation_set or generator has to be set, so that the “TensorBoardCallbacksCaller”
                               can execute the model on data to perform the evaluation.
        :param generator: Generator suitable to be passed to the model.
                          Either validation_set or generator has to be set, so that the “TensorBoardCallbacksCaller”
                          can execute the model on data to perform the evaluation.
        :param unlabel_func: If not None, this function is used to transform the outputs of the model back to concrete
                             goes flux values.
        :param verbose: Determines if additional debugging data, such as all true and predicted labels, should be
                        printed to the console.
        :param callbacks: List of functions, conforming to f(original_callback, epoch, y_true, y_pred).
                          These are all of the custom metrics which are executed after the initialization of the model,
                          after every epoch and after training has completed.
        """
        super().__init__()
        assert (validation_set is None) != (generator is None), \
            "Specify either validation_set or generator (and not both)"

        callbacks.append(adjustedR2FuncFactory(generator.dim if validation_set is None else validation_set[0].shape)) # generator cannot be used in default parameter

        self.predict_gen_params = predict_gen_params
        self.generator = generator
        self.validation_set = validation_set
        self.tag = "Callbacks Caller"
        self.unlabel_func = unlabel_func
        self.verbose = verbose
        self.callbacks = callbacks
        self.dir = tensorBoard.log_dir
        self.tensorBoard = tensorBoard

    def add_simple_value(self, name, value, epoch):
        if self.verbose:
            print(f'{name}: {value}')

        summary=tf.Summary()
        summary.value.add(tag=name, simple_value = value)
        self.tensorBoard.writer.add_summary(summary, epoch)

    def on_epoch_end(self, epoch, logs={}):
        if self.generator is not None:
            # prepare true labels
            y_true = self.generator.get_labels()

            # prepare predicted labels
            y_pred = self.model.predict_generator(generator=self.generator, **(self.predict_gen_params), verbose=True)
        else:
            # prepare true labels
            y_true = self.validation_set[1]

            # prepare predicted labels
            y_pred = self.model.predict(self.validation_set[0], **self.predict_gen_params, verbose=True)


        if self.unlabel_func is not None:
            y_pred = self.unlabel_func(y_pred)
        y_pred = y_pred.flatten()

        if (self.verbose):
            print("Custom Callbacks: Caller:")

            print("True Labels:")
            print(y_true)

            print("Predicted Labels:")
            print(y_pred)

        for callback in self.callbacks:
            callback(self, epoch, y_true, y_pred)

        self.tensorBoard.writer.flush()

        return
