import tensorflow as tf
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model,Sequential
from keras.layers import Dense,BatchNormalization
# from keras import backend as K
import numpy as np
# from datagen_train import DataGenerator as DataGenerator_train
from datagen_train_1536 import DataGenerator as DataGenerator_train_1536
# from datagen_predict import DataGenerator as DataGenerator_predict
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import time
import pickle
# def BaejiNet_temporal_stream():
#     X_input = Input(shape=(299,299,10))
#     x_corrected = Conv2D(3,(3,3),padding = 'same')(X_input)
#     base = InceptionResNetV2(weights='imagenet',include_top=False)(x_corrected)
#     #model = Model(InputModel.input,base(InputModel.output))
#     #x =  Flatten()(base)
#     x = Dense(512,activation = 'relu')(base)
#     output = Dense(4,activation = 'softmax')(x)
#     model = Model(inputs = X_input,outputs = output)
#     return model

#
# def BaejiNet_spatial_stream():
# 	base = InceptionResNetV2(weights='imagenet',input_shape = (299,299,3),include_top=False)
# 	x = base.output
# 	x = GlobalAveragePooling2D()(x)
# 	# x = Dense(512,activation = 'relu', input_shape = ())
# 	# input = Input(shape = (1536,))
# 	# x = Dense(512,activation = 'relu')(input)
# 	# output = Dense(4,activation = 'softmax')(x)
# 	model = Model(inputs =base.input ,outputs = x)
# 	return model

model2 = Sequential()
# model2.add(Dense(512,activation='relu',input_shape = (1536,)))
model2.add(BatchNormalization(input_shape = (1536,)))
model2.add(Dense(5,activation='softmax'))




params = {'batch_size': 32,
          'n_classes': 5,
          'shuffle': True}

# Datasets
frames = np.load('dataset_x_allgo_new.npy',allow_pickle=True)
# labels = np.load('dataset_y_new.npy',allow_pickle=True)


# labels = labels.item()
np.random.seed(10)
np.random.shuffle(frames)
# frames = frames[:100]
fc = len(frames)
frames_train = frames[:int(fc*0.75),:]
frames_val = frames[int(fc*0.75):int(fc*0.9),:]
# frames_test = frames[int(fc*0.9):,:]
print(frames_train.shape)
print(frames_val.shape)
# print(frames_test.shape)
# Generators
# training_generator = DataGenerator_train(frames_train, labels, **params)
training_generator = DataGenerator_train_1536(frames_train, **params)
validation_generator = DataGenerator_train_1536(frames_val, **params)
# predict_generator = DataGenerator_predict(frames, **params)

# model = BaejiNet_spatial_stream()

# for layer in model.layers[:-1]:
#    layer.trainable = False
# for layer in model.layers[-1:]:
#    layer.trainable = True

model2.load_weights('./models/weights-improvement-1730-0.89.hdf5')
print('loaded_weights')
# model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])
model2.compile(optimizer = Adam(lr=0.0002),loss = 'categorical_crossentropy',metrics=['acc'])
# model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,clipvalue=0.5 ,amsgrad=False), loss='categorical_crossentropy',metrics=['acc'])
# print(model.summary())

filepath = "./models/weights-final-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only = True ,save_best_only=True, mode='max',period = 10)
callbacks_list = [checkpoint]

start = time.time()


model2.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = 2000,
                    callbacks = callbacks_list,
                    verbose = 1,
                    use_multiprocessing=True,
                    workers= 6 )


print(time.time()-start)
