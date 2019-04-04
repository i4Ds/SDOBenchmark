from model_utils.model_runner import ModelRunner
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Dense, Flatten, Activation, concatenate

mr = ModelRunner('sfp_alexnet_v1')  # , cuda_visible_devices="2", log_device_placement=False)

main_input = mr.init_input( #sample_data_only = True,
              include_date  = True,
              channels      = ['magnetogram', '131', '1700'])

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
layers = Dense(1001, activation='softmax')(layers)

layers = Dense(1, activation='linear')(layers)

mr.train(layers) # dont_train=True)
