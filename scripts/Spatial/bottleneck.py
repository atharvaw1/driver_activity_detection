import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
# from datagen_train import DataGenerator as DataGenerator_train
from datagen_predict_bottleneck import DataGenerator as DataGenerator_predict
from keras import metrics
from keras.callbacks import ModelCheckpoint
import time
import pickle



def BaejiNet_spatial_stream():
	base = InceptionResNetV2(weights='imagenet',input_shape = (299,299,3),include_top=False)
	x = base.output
	output = GlobalAveragePooling2D()(x)
    #x = Dense(512,activation = 'relu')(x)
    #output = Dense(4,activation = 'softmax')(x)



	model = Model(inputs = base.input,outputs = output)
	return model




params = {'batch_size': 1,
          'n_classes': 5,
          'shuffle': False}

# Datasets
frames = np.load('dataset_x_allgo.npy',allow_pickle=True)
# frames = frames[:50]
print(frames.shape)
# exit()
# np.random.shuffle(frames)
# frames = frames[:100]
# np.save('dataset_test',frames)
# fc = len(frames)
model = BaejiNet_spatial_stream()
model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['acc'])
# Generators for


# size = 10000
size_batch = 10
for i in range(size_batch):
	print((len(frames)//size_batch)*i,(len(frames)//size_batch)*(i+1))
	predict_generator = DataGenerator_predict(frames[(len(frames)//size_batch)*i:(len(frames)//size_batch)*(i+1)], **params)
	out = model.predict_generator(generator=predict_generator, verbose=1)
# out = np.array(out)
	print(out.shape,frames.shape)
# print(out)
	out = np.concatenate((frames[(len(frames)//size_batch)*i:(len(frames)//size_batch)*(i+1)],out),axis=1)
	# print(out)
	print(out.shape)
	np.save("bottleneck_files/{0}".format(27+i),out)
	print(i)
# for i in range(int(fc/size)+1):
# 	try:
# 		predict_generator = DataGenerator_predict(frames[i*size:(i+1)*size], **params)
# 	except IndexError:
# 		predict_generator = DataGenerator_predict(frames[i*size:], **params)
#
#
# 	out = model.predict_generator(generator=predict_generator, verbose=1)
#
# 	try:
# 		out = np.concatenate((frames[i*size:(i+1)*size],out),axis=1)
# 	except:
# 		try:
# 			out = np.concatenate((frames[i*size:],out),axis=1)
# 		except:
# 			pass
	# print(out.shape)
	# print(frames.shape,out.shape)

	# f.seek(0,2)






#print(model.summary())


# start = time.time()

# print(time.time()-start)
