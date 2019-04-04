from keras.layers import Dense, Flatten, TimeDistributed, Conv2D, Activation, MaxPooling2D
from keras.optimizers import Adam

from model_utils.model_runner import ModelRunner, get_label_funcs_threshold

mr = ModelRunner('simple_cnn_model_threshold')  # cuda_visible_devices="", log_device_placement=False

bin_count = 10
main_input = mr.init_input(augment_data=False, sample_data_only=True,
                           label_funcs=get_label_funcs_threshold(bin_count))
 # Please consider the comments about encoding functions in model_utils/model_runner.py starting on line 37 before running this modell.

layers = TimeDistributed(Conv2D(8, (5, 5), strides=(2, 2)))(main_input)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))(layers)
layers = Flatten()(layers)
layers = Dense(bin_count - 1, activation='sigmoid', name='main_output')(layers)

mr.train(layers, epochs=5, batch_size=1, optimizer=Adam())
