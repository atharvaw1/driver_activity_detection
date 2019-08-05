import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, AveragePooling2D,Input,Flatten,Conv2D,BatchNormalization,GlobalAveragePooling2D
from keras import backend as K
import numpy as np
# from datagen_train_1536 import DataGenerator as DataGenerator_train_1536
from keras import metrics
from datagen_optical import DataGenerator as dg
from sklearn.metrics import classification_report,confusion_matrix


frames= np.load("gowda_optical_smoking.npy",allow_pickle=True)
anchor=50

prvs =cv.resize(frames[anchor],(299,299)) 
hsv = np.zeros_like(prvs)
X = np.empty((299,299,9))

prvs = cv.cvtColor(prvs,cv.COLOR_BGR2GRAY)
flow_stack = np.empty([299,299,0])
hsv[...,1] = 255  
for z in range(9):
    next = cv.resize(frames[anchor+z+1],(299,299))
    next = cv.cvtColor(next,cv.COLOR_BGR2GRAY)
    prvs = cv.medianBlur(prvs,5)
    next = cv.medianBlur(next,5)
    # print(prvs.shape,next.shape)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5,3,7,4,7,5, 0)
    # flow_stack.append(flow) # 299,299,2,10
    # flow_stack = np.concatenate((flow_stack,flow),axis=-1)
    prvs = next
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    mag = (mag>1) * mag
    hsv[...,0] = ang*180/np.pi/2                                     #hue   which colour  draw for purple and yellow
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)                # intensity brightness draw for extra bright
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    bgr = cv.medianBlur(bgr,5)
    gray = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    gray = np.reshape(gray , [299,299,1])
    flow_stack = np.concatenate((flow_stack,gray),axis =2)
    cv.imshow('frames',gray)
    cv.imshow('frames2',next)
    cv.waitKey(20)
# print(X[i,].shape, flow_stack.shape)
X = flow_stack
cv.destroyAllWindows()


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

model2 = BaejiNet_temporal_stream()

model2.load_weights('./models/weights-improvement-04-0.89.hdf5')
print('Loaded weights')
model2.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])
X = np.reshape(X,(1,299,299,9))
out = model2.predict(X)

print(np.argmax(out)+1)