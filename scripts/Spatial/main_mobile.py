import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,Flatten,BatchNormalization,Conv2D,AveragePooling2D
from keras import backend as K
import numpy as np
from datagen_train import DataGenerator as DataGenerator_train
# from datagen_predict_bottleneck import DataGenerator as DataGenerator_predict
from keras import metrics
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import time
import pickle



# def BaejiNet_spatial_stream():
# 	# x = Input(shape=(224,224,3))
# 	base = MobileNetV2(weights='imagenet',input_shape = (224,224,3),include_top=False)
# 	x = base.output
# 	x = GlobalAveragePooling2D()(x)
# 	# x = BatchNormalization()(x)
# 	# x = Dense(512,activation = 'relu')(x)
# 	output = Dense(5,activation = 'softmax')(x)
# 	# output = Flatten()(x)
# 	model = Model(inputs = base.input,outputs = output)
# 	return model

def BaejiNet_spatial_stream():
    X_input = Input(shape=(224,224,3))
    x = Conv2D(6,(5,5),padding = 'valid',activation='relu')(X_input)
    x = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)
    x = Conv2D(16, kernel_size=(5, 5), activation='relu', padding='valid')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = Flatten()(x)
    # x = Conv2D(120, kernel_size=(5, 5), activation='relu', padding='valid')(x)
    # x = GlobalAveragePooling2D()(x)
    # x = BatchNormalization()(x)
    x = Dense(120,activation='relu')(x)
    x = Dense(84,activation='relu')(x)
    output = Dense(5,activation='softmax')(x)
    # base = InceptionResNetV2(weights='imagenet',include_top=False)(x_corrected)
    #model = Model(InputModel.input,base(InputModel.output))
    #x =  Flatten()(base)
    #x = Dense(512,activation = 'relu')(base)
    # x = GlobalAveragePooling2D()(x_corrected)
    # output = Dense(2,activation = 'softmax')(x)
    model = Model(inputs = X_input,outputs = output)
    return model




params = {'batch_size': 64,
          'n_classes': 5,
          'shuffle': True}

# Datasets
frames = np.load('allgo_ir_index.npy',allow_pickle=True)
np.random.seed(10)
np.random.shuffle(frames)
# frames = frames[:100]
fc = len(frames)
print(fc)
frames_train = frames[:int(fc*0.75)]
frames_val = frames[int(fc*0.75):int(fc*0.9)]
ft = len(frames_train)
fv = len(frames_val)
print(frames_train.shape,frames_val.shape)

# frames_test = frames[int(fc*0.9):,:]
# print(frames.shape)



model = BaejiNet_spatial_stream()
# model.load_weights('./models_mobile/old-weights-final-v2-05-0.28.hdf5')
for layer in model.layers:
   layer.trainable = True
# for layer in model.layers[-1:]:
   # layer.trainable = True

model.compile(optimizer = 'Adam' ,loss = 'categorical_crossentropy',metrics=['acc'])
# print(model.summary())




# Generators for
training_generator = DataGenerator_train(frames_train, **params)
validation_generator = DataGenerator_train(frames_val, **params)


filepath = "./models_mobile/weights-final-v3-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only = True ,save_best_only=True, mode='max',period = 1)
callbacks_list = [checkpoint]

start = time.time()
# model.predict_generator(generator=training_generator,verbose=1)
#
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = 50,
                    callbacks = callbacks_list,
                    verbose = 1,
                    use_multiprocessing=True,
                    workers= 6 )

print(time.time()-start)
# size = 10000
# size_batch = 10
# for i in range(size_batch):
# 	print((len(frames)//size_batch)*i,(len(frames)//size_batch)*(i+1))
# 	predict_generator = DataGenerator_predict(frames[(len(frames)//size_batch)*i:(len(frames)//size_batch)*(i+1)], **params)
# 	out = model.predict_generator(generator=predict_generator, verbose=1)
# # out = np.array(out)
# 	print(out.shape,frames.shape)
# # print(out)
# 	out = np.concatenate((frames[(len(frames)//size_batch)*i:(len(frames)//size_batch)*(i+1)],out),axis=1)
# 	# print(out)
# 	print(out.shape)
# 	np.save("bottleneck_files/{0}".format(27+i),out)
# 	print(i)
