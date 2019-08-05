import numpy as np
import keras
import pickle
import cv2 as cv


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_frames, batch_size=32,n_classes=4, shuffle=True):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size

        self.list_frames = list_frames
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_frames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_frames_temp = [self.list_frames[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_frames_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_frames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_frames_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,1536))

        # Generate data
        for i, video in enumerate(list_frames_temp):
            if video[1]=='-1':
                X[i,] = np.load('bottleneck_data_sorted/'+video[0][16:]+'.npy' , allow_pickle=True)
            else:
                # Store sample
                X[i,] = np.load('bottleneck_data_sorted/'+video[0][16:]+video[1]+'.npy' , allow_pickle=True)

        return X
