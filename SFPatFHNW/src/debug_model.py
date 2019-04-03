import datetime as dt
import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical

from evaluation.callbacks_caller import TensorBoardCallbacksCaller

USE_REGRESSION = True
model_name = f"debug_model"
base_path = "D:/FHNW/semester_5/IP5/SFPatFHNW/data/sample"
# base_path = "../data/sample"

# load csv
train_data = pd.read_csv(os.path.join(base_path, "train", "meta_data.csv"), sep=',', parse_dates=["start", "end"])
test_data = pd.read_csv(os.path.join(base_path, "test", "meta_data.csv"), sep=',', parse_dates=["start", "end"])

# load data
channel = "magnetogram"
time_offset = dt.timedelta(minutes=10)  # time before end
tolerance = dt.timedelta(minutes=15)  # consider images with time +/- tolerance

regex_filename = re.compile(r'(?P<date>[^_]+)__(?P<channel>.*)\.jpg')
regex_id = re.compile(r'(?P<patch>\d+)_(?P<date>.*)')


def load_data(directory: str, data):

    X = np.full((len(data.index), 256, 256, 1), 127)
    y = np.array(data['peak_flux'])

    for sample in data.itertuples():
        match = regex_id.match(sample.id)
        if match:
            # check if the label is correct
            assert y[sample.Index] == sample.peak_flux

            # load image
            path = os.path.join(directory, match.group("patch"), match.group("date"))
            for file in os.listdir(path):
                match = regex_filename.match(file)
                if match and match.group('channel') == channel:
                    img_datetime = dt.datetime.strptime(match.group('date'), "%Y-%m-%dT%H%M%S")
                    if abs(sample.end - img_datetime) < tolerance:
                        X[sample.Index, :, :] = img_to_array(load_img(os.path.join(path, file), color_mode="grayscale"))
                        break
        else:
            print(f"problem with sample: {sample}")
    return X, y


X_train, y_train = load_data(os.path.join(base_path, "train"), train_data)
X_val, y_val = load_data(os.path.join(base_path, "test"), test_data)

# output a min flux and a max flux example image
minIndex = np.argmin(y_train)
maxIndex = np.argmax(y_train)
plt.imshow(np.squeeze(X_train[minIndex], 2))
plt.title(f"peak flux: {y_train[minIndex]}")
plt.savefig("min.png")

plt.title(f"peak flux: {y_train[maxIndex]}")
plt.imshow(np.squeeze(X_train[maxIndex], 2))
plt.savefig("max.png")

# Label encode/decode functions
LabelFuncs = namedtuple('LabelFuncs', ['label', 'unlabel'])


def get_label_funcs(n_classes=10, lower=1e-9, upper=1e-3):
    low = np.log10(lower)
    up = np.log10(upper)

    def label_func(y):
        val = np.log10(y)
        rawbin = (val - low) / (up - low) * n_classes
        index = np.floor(rawbin).astype(int)
        return to_categorical(np.clip(index, 0, n_classes - 1), n_classes)

    def unlabel_func(y):
        val = np.argmax(y, axis=-1)
        exponent = (((val / n_classes) * (up - low)) + low) #SF: +0.5?
        return 10 ** exponent

    return LabelFuncs(label_func, unlabel_func)


def get_label_funcs_bin(threshold):
    def label_func(y):
        return np.asarray([[0, 1] if a > threshold else [1, 0] for a in y])

    def unlabel_func(y):
        return np.asarray([1e-9 if np.argmax(a, -1) == 0 else 1e-5 for a in y])

    return LabelFuncs(label_func, unlabel_func)


CLASSES = 2
EPOCHS = 5

# label_encode = get_label_funcs(CLASSES, min(y_train), max(y_train))
mean_flux = 5.2941e-07
label_encode = get_label_funcs_bin(mean_flux)
y_train_encoded = y_train if USE_REGRESSION else label_encode.label(y_train)
y_val_encoded = y_val if USE_REGRESSION else label_encode.label(y_val)

# model
images_input = Input(shape=(256, 256, 1), name='images_input')
layers = Conv2D(8, (5, 5), strides=(2, 2), activation="relu")(images_input)
layers = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(layers)
layers = Flatten()(layers)
main_output = Dense(1 if USE_REGRESSION else CLASSES, activation='linear' if USE_REGRESSION else 'sigmoid',
                    name='main_output')(layers)

model = Model(inputs=[images_input], outputs=[main_output])
model.compile(optimizer=Adam(), loss='mean_absolute_error' if USE_REGRESSION else 'binary_crossentropy')
model.summary()

# ----------------
# set up callbacks
# ----------------
callbacks = []

now_formatted = dt.datetime.now().strftime('%d.%m.%Y_%H.%M.%S')

# Ensure required directories exist
if not os.path.isdir('../logs'):
    os.makedirs('../logs')
if not os.path.isdir('../models_trained'):
    os.makedirs('../models_trained')

log_dir = '../logs/{}/{}'.format(model_name, now_formatted) + ("_reg" if USE_REGRESSION else "_class")
callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True))

callbacks.append(TensorBoardCallbacksCaller(callbacks[0], {}, validation_set=(X_val, y_val), unlabel_func=None if USE_REGRESSION else label_encode.unlabel, verbose=True))

# ---------
# train
# ---------
def predict():
    y_pred_encoded = model.predict(X_val)
    y_pred = y_pred_encoded if USE_REGRESSION else label_encode.unlabel(y_pred_encoded)

    print("Predictions encoded")
    print(y_pred_encoded)
    print("Predictions decoded")
    print(y_pred)
    mae = np.sum(np.abs(y_val - y_pred)) / len(y_val)
    print(f"mae: {mae}")


print("\nPredict before training:")
predict()

history = model.fit(X_train, y_train_encoded, validation_data=(X_val, y_val_encoded), epochs=EPOCHS, callbacks=callbacks)

print("\nPredict after training:")
predict()
