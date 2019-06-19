import tensorflow as tf
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization
from keras import backend as K
import numpy as np
from keras.optimizers import SGD,Adam


def NNet_spatial_stream():
	input = Input(shape=(1536,))
	x = BatchNormalization()(input)
	output = Dense(4,activation = 'softmax')(x)
	model = Model(inputs =input ,outputs = output)
	return model


model = NNet_spatial_stream()


model.compile(optimizer=SGD(lr=0.0003	), loss='categorical_crossentropy',metrics=['acc'])

video = np.load('dataset_x_improved.npy',allow_pickle=True)
labels = np.load('dataset_y.npy',allow_pickle=True)
labels = labels.item()
np.random.shuffle(video)
video = video[:100]
X = np.empty((len(video) ,1536))
y  = np.empty((len(video),4))
for i,v in enumerate(video):
	X[i,] = np.load('bottleneck_data_sorted/'+v[0][16:]+v[1]+'.npy' , allow_pickle=True)
	y[i] = labels[v[0]]
# X = np.reshape(X,(1,1536))
print(X.shape)
# m = np.mean(X,axis=1)
# std = np.std(X,axis=1)
# print(m.shape)
# print(std.shape)
# X = (X-np.vstack(m))/(np.vstack(std))
# print(X_new)
# X = np.array([[0,0],[1,0],[0,1],[1,1]])
# y = np.array([[1,0,0,0]])

model.fit(X,y,epochs=1000)
# print(model.predict(X))
