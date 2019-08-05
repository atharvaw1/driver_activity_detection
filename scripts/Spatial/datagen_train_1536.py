import numpy as np
import keras
import pickle



class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_frames, batch_size=32,n_classes=4, shuffle=True):
		'Initialization'
		self.batch_size = batch_size
		# self.labels = labels
		self.list_frames = list_frames
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
		X, y = self.__data_generation(list_frames_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_frames))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_frames_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size,1536))
		y = np.empty((self.batch_size,5))


		# Generate data
		for i, video in enumerate(list_frames_temp):
			# if video[1]=='-1':
			X[i,] = np.load('bottleneck_data_sorted2/'+video[0][21:-4]+video[1]+'.npy' , allow_pickle=True)
			if video[0][21]=='D':
				y[i] = [1,0,0,0,0]
			elif video[0][21]=='T':
				y[i] = [0,0,0,1,0]
			# else:
				# Store sample
				# X[i,] = np.load('bottleneck_data_sorted2/'+video[0][16:]+video[1]+'.npy' , allow_pickle=True)
				# Store class
				# y[i] = self.labels[video[0]]
		return X,y
