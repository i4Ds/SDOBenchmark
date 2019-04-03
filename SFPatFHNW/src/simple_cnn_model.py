from keras.optimizers import Adam

from model_utils.model_runner import ModelRunner
from keras.layers import Dense, Flatten, TimeDistributed, Conv2D, Activation, MaxPooling2D

mr = ModelRunner('simple_cnn_model')  # , cuda_visible_devices="2", log_device_placement=False)

main_input = mr.init_input(augment_data=False, sample_data_only=False)

layers = TimeDistributed(Conv2D(8, (5, 5), strides=(2, 2)))(main_input)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))(layers)
layers = Flatten()(layers)
layers = Dense(1, activation='linear')(layers)

mr.train(layers, optimizer=Adam(), epochs=50, batch_size=1)
