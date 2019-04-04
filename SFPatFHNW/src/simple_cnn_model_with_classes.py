from model_utils.model_runner import ModelRunner, get_label_funcs_onehot
from keras.layers import Dense, Flatten, TimeDistributed, Conv2D, Activation, MaxPooling2D

mr = ModelRunner('simple_cnn_model_with_classes') # cuda_visible_devices="", log_device_placement=False

classes_count = 10
main_input = mr.init_input(augment_data  = False, sample_data_only = True,
              label_funcs = get_label_funcs_onehot(classes_count))
# Please consider the comments about encoding functions in model_utils/model_runner.py starting on line 37 before running this modell.

layers = TimeDistributed(Conv2D(8, (5, 5), strides=(2, 2)))(main_input)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))(layers)
layers = Flatten()(layers)
layers = Dense(classes_count, activation='softmax', name='main_output')(layers)

mr.train(layers, epochs=5, batch_size=1)
