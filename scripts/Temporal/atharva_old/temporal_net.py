import tensorflow as tf
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, AveragePooling2D,Input,Flatten,Conv2D,BatchNormalization,GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from keras import metrics
from keras.callbacks import ModelCheckpoint
from datagen_optical import DataGenerator as dg


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
    # base = InceptionResNetV2(weights='imagenet',include_top=False)(x_corrected)
    #model = Model(InputModel.input,base(InputModel.output))
    #x =  Flatten()(base)
    #x = Dense(512,activation = 'relu')(base)
    # x = GlobalAveragePooling2D()(x_corrected)
    # output = Dense(2,activation = 'softmax')(x)
    model = Model(inputs = X_input,outputs = output)
    return model


# def BaejiNet_spatial_stream():

#     base = InceptionResNetV2(weights='imagenet',input_shape = (299,299,3),include_top=False)
#     x = base.output
#     x = GlobalAveragePooling2D()(x)
#     # x = Dense(512,activation = 'relu')(x)
#     output = Dense(4,activation = 'softmax')(x)



#     model = Model(inputs = base.input,outputs = output)
#     return model




params = {'dim': (224,224),
          'batch_size': 16,
          'n_classes': 2,
          'n_channels': 9,
          'shuffle': True}

# Datasets
frames = np.load('dataset_x_temporal+rr.npy',allow_pickle=True)
# labels = np.load('dataset_y.npy',allow_pickle=True)
fc = len(frames)
print(fc)
#exit()
# labels = labels.item()
np.random.seed(10)
np.random.shuffle(frames)
#frames = frames[:100]
frames_train = frames[:int(fc*0.75),:]
frames_val = frames[int(fc*0.75):int(fc*0.9),:]
print(frames_train.shape,frames_val.shape)
# Generators
# training_generator = DataGenerator(frames_train, labels, **params)
# validation_generator = DataGenerator(frames_val, labels, **params)

training_generator = dg(frames_train, **params)
validation_generator = dg(frames_val, **params)


model = BaejiNet_temporal_stream()
model.load_weights('./models/weights-improvement-01-0.83.hdf5')
print('Loaded weights')
model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])

print(model.summary())

filepath = "./models/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_weights_only = True ,save_best_only=True, mode='max',period = 1)
callbacks_list = [checkpoint]

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = 2000 ,
                    # steps_per_epoch = int(np.floor(len(frames_train) / params['batch_size'])),
                    callbacks = callbacks_list,
                    verbose = 1,
                    use_multiprocessing=True,
                    workers= 6 )
