import numpy as np
import keras
import pickle
import cv2 as cv
import re

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_frames, batch_size=32, dim=(299,299), n_channels=10,
                 n_classes=4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_frames = list_frames
        self.n_channels = n_channels
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
        X= self.__data_generation(list_frames_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_frames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_frames_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,self.n_classes))


        # Generate data
        for i, video in enumerate(list_frames_temp):
            # Store sample

            if video[0][2] == 'S':
                with open(video[0],'rb') as f1:
                    frames = pickle.load(f1)
                    anchor = int(video[1])-5
                y[i] = np.array([1,0])  #stair action
            else:
                frames = []
                if video[0][19] == 'A':
                    y[i] = np.array([1,0]) #allgo action
                else:
                    y[i] = np.array([0,1]) #allgo no action
                if video[0][-1]=='r':
                    for j in range(int(video[1])+4,int(video[1])-6,-1):
                        x = re.search('_[0-9]+\.',video[0])
                        path = video[0][:x.start()+1] + str(j) + '.jpg'
                        frames.append(cv.imread(path))
                        # print(path)
                    # print(np.array(frames).shape)
                    # anchor = int(video[1])+5
                else:
                    for j in range(int(video[1])-5,int(video[1])+5):
                        x = re.search('_[0-9]+\.',video[0])
                        path = video[0][:x.start()+1] + str(j) + '.jpg'
                        frames.append(cv.imread(path))
                        # print(path)
                    # print(np.array(frames).shape)
                    # anchor = int(video[1])-5
                anchor = 0
            
            prvs = frames[anchor]
            hsv = np.zeros_like(prvs)
            try:
                prvs = cv.cvtColor(prvs,cv.COLOR_BGR2GRAY)
            except:
                print(frames,anchor,video)
                exit()
            flow_stack = np.empty([299,299,0])
            hsv[...,1] = 255  
            for z in range(9):
                # prvs = frames[anchor+z]
                try:
                    next = frames[anchor+z+1]
                except:
                    print(video,anchor,np.array(frames).shape)
                    exit()
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
                #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           print(flow_stack.shape)
            # print(X[i,].shape, flow_stack.shape)
            X[i,] = flow_stack
                                                                                                                                                                                                                                                                                                                                                                
            # Store class
            

            #y = np.array([0,0,1,0])
        # print(y.shape)
        # print(X.shape)

        return X



# next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
#     prvs = cv.medianBlur(prvs,5)
#     next = cv.medianBlur(next,5)
#     print(prvs.shape,next.shape)
#     flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5,3,7,4,7,5, 0)
