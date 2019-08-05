import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, AveragePooling2D,Input,Flatten,Conv2D,BatchNormalization,GlobalAveragePooling2D
from keras import backend as K
import numpy as np
# from datagen_train_1536 import DataGenerator as DataGenerator_train_1536
from keras import metrics
from datagen_optical import DataGenerator as dg
from sklearn.metrics import classification_report,confusion_matrix


def BaejiNet_temporal_stream():
    X_input = Input(shape=(299,299,9))
    x = Conv2D(6,(5,5),padding = 'valid',activation='relu')(X_input)
    x = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)
    x = Conv2D(16, kernel_size=(5, 5), activation='relu', padding='valid')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = Conv2D(120, kernel_size=(5, 5), activation='relu', padding='valid')(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(84,activation='relu')(x)
    output = Dense(2,activation='softmax')(x)
    model = Model(inputs = X_input,outputs = output)
    return model

frames = np.load('dataset_x_temporal+rr.npy',allow_pickle=True)
# labels = np.load('dataset_y_new.npy',allow_pickle=True)
# labels = labels.item()
np.random.seed(10)
np.random.shuffle(frames)

fc= len(frames)

frames_test = frames[int(fc*0.9):,:]
# frames_test = frames_test[:100]
print(frames_test.shape)

params = {'dim': (299,299),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 9,
          'shuffle': False}

predict_generator = dg(frames_test, **params)

model2 = BaejiNet_temporal_stream()

model2.load_weights('./models/weights-improvement-04-0.89.hdf5')
print('Loaded weights')
model2.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])

out = model2.predict_generator(generator=predict_generator,verbose=1)
print(out.shape)
y_pred = []
y_true = []
for i,video in enumerate(frames_test):
	y_pred.append(np.argmax(out[i])+1)
	if video[0][2]=='S' or video[0][19]=='A':
		y_true.append(1)
	else:
		y_true.append(2)
		# print(video[0])
# print(y_pred)
# print(y_true)
# print(set(y_true)-set(y_pred))
print(classification_report(y_true,y_pred,target_names = ['Action','No action']))
print(confusion_matrix(y_true,y_pred))
# print(out,np.sum(out,axis=1))
