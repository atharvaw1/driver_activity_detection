import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Dense,BatchNormalization
from keras import backend as K
import numpy as np
# from datagen_train_1536 import DataGenerator as DataGenerator_train_1536
from keras import metrics
from datagen_predict import DataGenerator as DataGenerator_predict
from sklearn.metrics import classification_report,confusion_matrix


frames = np.load('dataset_x_improved.npy',allow_pickle=True)[:120000]
labels = np.load('dataset_y_new.npy',allow_pickle=True)
labels = labels.item()
np.random.seed(10)
np.random.shuffle(frames)

fc= len(frames)

frames_test = frames#[int(fc*0.9):,:]
# frames_test = frames_test[:64]
print(frames_test.shape)

params = {'batch_size': 16,
		'n_classes': 5,
		'shuffle': False}
predict_generator = DataGenerator_predict(frames_test, **params)


model2 = Sequential()
# model2.add(Dense(512,activation='relu',input_shape = (1536,)))
model2.add(BatchNormalization(input_shape = (1536,)))
model2.add(Dense(5,activation='softmax'))
model2.load_weights('./models/weights-improvement-1730-0.89.hdf5')
print('Loaded weights')
model2.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])

out = model2.predict_generator(generator=predict_generator,verbose=1)
print(out.shape)
y_pred = []
y_true = []
for i,video in enumerate(frames_test):
	y_pred.append(np.argmax(out[i])+1)
	if video[1]!='-1':
		y_true.append(np.argmax(labels[video[0]])+1)
	else:
		y_true.append(5)
		# print(video[0])
# print(y_pred)
# print(y_true)
# print(set(y_true)-set(y_pred))
print(classification_report(y_true,y_pred,target_names = ['Drinking','Eating_snack','Smoking','Telephoning','Safe']))
print(confusion_matrix(y_true,y_pred))
# print(out,np.sum(out,axis=1))
