from model_utils.model_runner import ModelRunner, get_label_funcs_onehot
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Dense, Flatten, Activation, concatenate

mr = ModelRunner('sfp_alexnet_onehot_v1') # cuda_visible_devices="", log_device_placement=False

classes_count = 1000
main_input = mr.init_input( #sample_data_only = True,
              include_date  = True,
              channels      = ['magnetogram', '131', '1700'],
              label_funcs    = get_label_funcs_onehot(classes_count))
# Please consider the comments about encoding functions in model_utils/model_runner.py starting on line 37 before running this modell.

layers = TimeDistributed(Conv2D(96, (11, 11), strides=(4, 4)))(main_input)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))(layers)

layers = TimeDistributed(Conv2D(256, (5, 5), padding='same'))(layers)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))(layers)

layers = TimeDistributed(Conv2D(384, (3, 3), padding='same'))(layers)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(Conv2D(384, (3, 3), padding='same'))(layers)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(Conv2D(256, (3, 3), padding='same'))(layers)
layers = TimeDistributed(Activation('relu'))(layers)
layers = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))(layers)

layers = Flatten()(layers)

layers = concatenate([layers, mr.get_date_input()])

layers = Dense(9217, activation='relu')(layers)
layers = Dense(4097, activation='relu')(layers)
layers = Dense(4097, activation='relu')(layers)

layers = Dense(classes_count, activation='softmax', name='main_output')(layers)

mr.train(layers) #, dont_train=True
