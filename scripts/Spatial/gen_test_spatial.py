import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Dense,BatchNormalization,GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from keras import metrics
from sklearn.metrics import classification_report,confusion_matrix
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import cv2 as cv

def BaejiNet_spatial_stream():
	base = InceptionResNetV2(weights='imagenet',input_shape = (299,299,3),include_top=False)
	x = base.output
	output = GlobalAveragePooling2D()(x)
	model = Model(inputs = base.input,outputs = output)
	return model
pic = cv.imread('Internet images/test.jpeg')
pic = cv.resize(pic,(299,299),fx=0,fy=0)
model1 = BaejiNet_spatial_stream()
model1.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])

model2 = Sequential()
# model2.add(Dense(512,activation='relu',input_shape = (1536,)))
model2.add(BatchNormalization(input_shape = (1536,)))
model2.add(Dense(5,activation='softmax'))
model2.load_weights('./models/weights-improvement-1730-0.89.hdf5')
print('Loaded weights')
model2.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])
pic = np.reshape(pic,(1,299,299,3))
print(pic.shape)
# print(model1.summary())
out1 = model1.predict(pic,verbose=1)
out = model2.predict([out1],verbose=1)
print(out)
labels = {1:'Drinking',2:'Eating_snack',3:'Smoking',4:'Tele',5:'Safe'}
print(labels[np.argmax(out)+1])
