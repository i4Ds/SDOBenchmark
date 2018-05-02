from skimage.io import imread
import keras.utils.data_utils
from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import os

class SDOBenchmarkGenerator(keras.utils.data_utils.Sequence):
    'Generates data for keras'
    def __init__(self, base_path, batch_size=32, dim=(256, 256, 4), shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.base_path = base_path
        self.dim = dim
        self.data = self.loadCSV()
        self.shuffle = shuffle
        self.on_epoch_end()

    def loadCSV(self):
        return pd.read_csv(os.path.join(self.base_path, 'train.csv'), sep=";", parse_dates=["start", "end", "peak"], index_col="id")

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [
            np.empty((self.batch_size, 1)),
            np.empty((self.batch_size, *self.dim))
        ]

        # Generate data
        data = self.data.iloc[indexes]
        X[0] = np.array((data['start'] - pd.Timestamp('2012-01-01 00:00:00')).astype(np.int64) // (24 * 3600 * 10 ** 9))
        X[0] /= (pd.Timestamp('2018-01-01 00:00:00') - pd.Timestamp('2012-01-01 00:00:00')).astype(np.int64) // (24 * 3600 * 10 ** 9)
        X[1] = np.array(map(self.loadImg, data.index))
        y = np.array(data['peak_flux'])

        return X, y

    def loadImg(self, sample_id):
        'Load the 4 images as 4 channels'
        ar_nr, p = sample_id.split("_", 1)
        path = os.path.join(self.base_path, ar_nr, p)

        slices = np.zeros(self.dim)
        i = 0
        for img in os.listdir(path):
            if img.endswith('_magnetogram.jpg'):
                slices[:,:,i] = img_to_array(load_img(path, grayscale=True))
                i += 1

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