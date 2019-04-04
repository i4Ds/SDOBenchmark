from model_utils.model_runner import ModelRunner
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, concatenate

mr = ModelRunner('sfp_vgg_transfer_v1')  # , cuda_visible_devices="2", log_device_placement=False)

main_input = mr.init_input( #sample_data_only = True,
              only_last_slice   = True,
              include_date      = True,
              channels          = ['magnetogram', '131', '1700'])

vgg16_pretrained = VGG16(weights='imagenet',
                         include_top=False,
                         input_tensor=main_input)
for layer in vgg16_pretrained.layers:
    layer.trainable = False

layers = Flatten()(vgg16_pretrained.layers[-1].output)
layers = concatenate([layers, mr.get_date_input()])

layers = Dense(4097, activation='relu')(layers)
layers = Dense(4097, activation='relu')(layers)
layers = Dense(1001, activation='softmax')(layers)

layers = Dense(1, activation='linear')(layers)

mr.train(layers) #, dont_train=True
