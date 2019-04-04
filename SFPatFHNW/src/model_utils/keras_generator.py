import datetime as dt
import os

import keras.utils.data_utils
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
import itertools

class SDOBenchmarkGenerator(keras.utils.data_utils.Sequence):
    'Generates data for keras \
    Inspired by https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html'

    def __init__(self, base_path, batch_size=32, dim=(4, 256, 256), channels=['magnetogram'], shuffle=True,
                 augment=True, label_func=None, data_format="channels_last", include_date=True, only_last_slice=False,
                 no_image_data=False, default_value=127.):
        'Initialization'
        self.batch_size = batch_size
        self.base_path = base_path
        self.data_format = data_format
        self.label_func = label_func
        self.dim = dim if len(dim) == 4 else (
            dim + (len(channels),) if data_format == 'channels_last' else (len(channels),) + dim)
        self.channels = channels
        self.include_date = include_date
        self.time_steps = [0, 7 * 60, 10 * 60 + 30, 11 * 60 + 50]
        self.data = self.loadCSV(augment)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.only_last_slice = only_last_slice
        self.no_image_data = no_image_data
        self.default_value = default_value

        self.imagesLoaded = 0
        self.imagesExpected = 0

    def loadCSV(self, augment=True):
        data = pd.read_csv(os.path.join(self.base_path, 'meta_data.csv'), sep=",", parse_dates=["start", "end"],
                           index_col="id")

        # augment by doubling the data and flagging them to be flipped horizontally
        data['flip'] = False
        if augment:
            new_data = data.copy()
            new_data.index += '_copy'
            new_data['flip'] = True
            data = pd.concat([data, new_data])
        return data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [
            np.empty((self.batch_size, 1))
        ]
        if self.include_date:
            X.append(np.empty((self.batch_size, *self.dim)))

        # Generate data
        data = self.data.iloc[indexes]
        X[0] = np.asarray(list(map(self.loadImg, data.itertuples())))

        ind = np.where(data['flip'])
        X[0][ind] = X[0][ind, ::-1, ...]

        if self.include_date:
            X[1] = (data['start'] - pd.Timestamp('2012-01-01 00:00:00')).view('int64')
            X[1] /= (pd.Timestamp('2018-01-01 00:00:00') - pd.Timestamp('2012-01-01 00:00:00')).view('int64')
        y = np.array(data['peak_flux'])
        if self.label_func is not None:
            y = self.label_func(y)
        return X, y

    def _label_generation(self, indexes):
        """same as __data_generation but only labels and not label_func"""
        data = self.data.iloc[indexes]
        return np.array(data['peak_flux'])

    def loadImg(self, sample):
        # initialize np arrays
        if self.only_last_slice:
            # ignore first dimension
            slices = np.full(self.dim[1:], self.default_value)
            self.imagesExpected += self.dim[3]
        else:
            slices = np.full(self.dim, self.default_value)
            self.imagesExpected += self.dim[0] * self.dim[3]

        if self.no_image_data:
            return slices

        # Load the images of each timestep as channels
        ar_nr, p = sample.Index.replace('_copy', '').split("_", 1)
        path = os.path.join(self.base_path, ar_nr, p)

        sample_date = sample.start
        time_steps = [sample_date + dt.timedelta(minutes=offset) for offset in self.time_steps]
        for img in [name for name in os.listdir(path) if name.endswith('.jpg')]:
            img_datetime_raw, img_wavelength = os.path.splitext(img)[0].split("__")
            img_datetime = dt.datetime.strptime(img_datetime_raw, "%Y-%m-%dT%H%M%S")

            # calc wavelength and datetime index
            datetime_index = [di[0] for di in enumerate(time_steps) if
                              abs(di[1] - img_datetime) < dt.timedelta(minutes=15)]
            if img_wavelength in self.channels and len(datetime_index) > 0:
                if not self.only_last_slice or datetime_index[0] == 3:
                    self.imagesLoaded += 1

                    val = np.squeeze(img_to_array(load_img(os.path.join(path, img), color_mode="grayscale")), 2)
                    if self.only_last_slice:
                        if self.data_format == 'channels_first':
                            slices[:, :, self.channels.index(img_wavelength)] = val
                        else:
                            slices[self.channels.index(img_wavelength), :, :] = val
                    else:
                        if self.data_format == 'channels_first':
                            slices[datetime_index[0], :, :, self.channels.index(img_wavelength)] = val
                        else:
                            slices[self.channels.index(img_wavelength), :, :, datetime_index[0]] = val

        return slices

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def get_labels(self):
        # Generate indexes of all batches
        indexes = [self.indexes[index * self.batch_size:(index + 1) * self.batch_size] for index in range(0,self.__len__())]
        indexes = list(itertools.chain.from_iterable(indexes))

        # Generate labels
        y = self._label_generation(indexes)

        return y
