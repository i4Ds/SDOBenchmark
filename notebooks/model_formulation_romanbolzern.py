'''
Current "competitive model" on kaggle. June 2017.
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
# limit memory usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
from keras.models import Model
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, concatenate, SeparableConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from notebooks.utils.keras_generator import SDOBenchmarkGenerator
import time
import numpy as np
#random.seed(a=1337)

base_path = '/media/all/D4/output/2012-01-01T000000_2018-01-01T000000/'

# Parameters
params = {'dim': (4, 256, 256, 4),
          'batch_size': 16,
          'channels': ['magnetogram', '304', '131', '1700'],
          'shuffle': True}

# category encoding quiet, A1..9, B1..9, ... X1..9, X10+
categories = to_categorical(np.arange(7))
params['label_func'] = lambda y: categories[np.floor(np.log10(y)+9).astype(int)]

# Generators
training_generator = SDOBenchmarkGenerator(os.path.join(base_path, 'training'), **params)
validation_generator = SDOBenchmarkGenerator(os.path.join(base_path, 'test'), **params)

category_balance = np.sum(params['label_func'](training_generator.data['peak_flux']), axis=0)
category_weights = np.sum(category_balance) / category_balance
category_weights[np.isinf(category_weights)] = 0
category_weights = dict(zip(range(len(categories)), category_weights))

start_time = time.time()

scales = [
    64, #256,
    64*3, #728,
    1, #8,
    128, #1024,
    256 #2048
]

#try:
images_input = Input(shape=params['dim'], name='images_input')

# https://stackoverflow.com/questions/43265084/keras-timedistributed-are-weights-shared
x = TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1'))(images_input)
x = TimeDistributed(BatchNormalization(name='block1_conv1_bn'))(x)
x = TimeDistributed(Activation('relu', name='block1_conv1_act'))(x)
x = TimeDistributed(Conv2D(64, (3, 3), use_bias=False, name='block1_conv2'))(x)
x = TimeDistributed(BatchNormalization(name='block1_conv2_bn'))(x)
x = TimeDistributed(Activation('relu', name='block1_conv2_act'))(x)


residual = TimeDistributed(Conv2D(scales[3], (1, 1), strides=(2, 2), padding='same', use_bias=False))(x)
residual = TimeDistributed(BatchNormalization())(residual)

x = TimeDistributed(Activation('relu', name='block13_sepconv1_act'))(x)
x = TimeDistributed(SeparableConv2D(scales[0]*3, (3, 3), padding='same', use_bias=False, name='block13_sepconv1'))(x)
x = TimeDistributed(BatchNormalization(name='block13_sepconv1_bn'))(x)
x = TimeDistributed(Activation('relu', name='block13_sepconv2_act'))(x)
x = TimeDistributed(SeparableConv2D(scales[3], (3, 3), padding='same', use_bias=False, name='block13_sepconv2'))(x)
x = TimeDistributed(BatchNormalization(name='block13_sepconv2_bn'))(x)

x = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool'))(x)
x = layers.add([x, residual])

x = TimeDistributed(SeparableConv2D(int(scales[3]/2*3), (3, 3), padding='same', use_bias=False, name='block14_sepconv1'))(x)
x = TimeDistributed(BatchNormalization(name='block14_sepconv1_bn'))(x)
x = TimeDistributed(Activation('relu', name='block14_sepconv1_act'))(x)

x = TimeDistributed(SeparableConv2D(scales[4], (3, 3), padding='same', use_bias=False, name='block14_sepconv2'))(x)
x = TimeDistributed(BatchNormalization(name='block14_sepconv2_bn'))(x)
x = TimeDistributed(Activation('relu', name='block14_sepconv2_act'))(x)

x = TimeDistributed(GlobalMaxPooling2D(name='max_pool'))(x)

x = Flatten()(x)

#x = Lambda(lambda x: 10. ** x)(x)

date_input = Input(shape=(1,), name='date_input')
x = concatenate([x, date_input])
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
main_output = Dense(len(categories), activation='softmax', name='main_output')(x)

def converted_mae(y_true, y_pred):
    conv_y_true = 10.**(K.cast(K.argmax(y_true, axis=-1), K.floatx())-9.) * 5.
    conv_y_pred = 10.**(K.cast(K.argmax(y_pred, axis=-1), K.floatx())-9.) * 5.
    #K.cast(xyz,K.floatx())
    return K.mean(K.abs(conv_y_pred - conv_y_true), axis=-1)

'''def tss(y_true, y_pred):
    y_true_geq_M = K.cast(K.greater_equal(K.cast(K.argmax(y_true, axis=-1), K.floatx()),5.), K.floatx())
    y_pred_geq_M = K.cast(K.greater_equal(K.cast(K.argmax(y_pred, axis=-1), K.floatx()),5.), K.floatx())

    # tss with special cases for 0 positive or 0 negative samples
    tp = K.sum(y_pred_geq_M * y_true_geq_M, axis=-1)
    tn = K.sum((1.-y_pred_geq_M) * (1.-y_true_geq_M), axis=-1)
    fp = K.sum((1.-y_true_geq_M) * y_pred_geq_M, axis=-1)
    fn = K.sum(y_true_geq_M * (1.-y_pred_geq_M), axis=-1)

    # Doesn't fit all fn

    return K.maximum(tp, K.epsilon()) / (K.maximum(tp + fn, K.epsilon())) - K.maximum(fp, K.epsilon()) / (K.maximum(fp + tn, K.epsilon()))'''

model = Model(inputs=[images_input, date_input], outputs=[main_output])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy', converted_mae])

model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1, min_lr=0.0000001) #, epsilon=1e-8
cb_tensorBoard = TensorBoard(log_dir='./logs/n1', histogram_freq=0, write_grads=True, write_graph=False, write_images=True)
cb_modelCheckpoint = ModelCheckpoint('./models/n1_' + os.environ['CUDA_VISIBLE_DEVICES'] + '_{epoch:02d}-{loss:.2f}-{val_loss:.2f}', monitor='val_loss', verbose=1, save_best_only=True)

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=len(training_generator),
                              validation_steps=len(validation_generator),
                              epochs=1000,
                              class_weight = category_weights,
                              callbacks=[cb_tensorBoard, reduce_lr, cb_modelCheckpoint], #reduce_lr
                              use_multiprocessing=True,
                              max_queue_size=params['batch_size'],
                              workers=params['batch_size'] * 2)