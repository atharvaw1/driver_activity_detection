import cv2
import numpy as np
import os
import pickle

# f1 = open('dataset_x_allgo','wb')
# d = pickle.load(f1)
# # f1.close()
d = []
file = 0
path = './Allgo_data_resized/'
for i in os.listdir(path):
	for j in os.listdir(path + i):
		# cap = cv2.VideoCapture('./SERRE/' + i + '/' + j)

		# fourcc = cv2.VideoWriter_fourcc(*'XVID')
		f1 = open(path+i+'/'+j,'rb')
		frames = pickle.load(f1)
		f1.close()
		# frame = cv2.imread('./SERRE/'+i+'/'+j)

		f=0
		video = []
		for frame in frames:
			# ret,frame = cap.read()
			# if not ret:
				# break
			b = cv2.resize(frame,(224,224),fx=0,fy=0) #interpolation = cv2.INTER_CUBIC)
			# print(b.shape)
			# video.append(b)
			d.append(["/home/atharva/DAD/Allgo_224/"+i+'/'+j + str(f)])
			cv2.imwrite("./Allgo_224/"+i+'/'+j+str(f)+'.png',b)
			# np.save("/home/atharva/DAD/STAIR_Lab_224/"+i+'/'+j+str(f) , b)
			f+=1
			# break
		file+=1
		if file%10 ==0 :
			print(file)

		# cap.release()
		# break
	# break
# f1.close()
d = np.array(d)
print(d,d.shape)
np.save('allgo_x_224',d)
# c=0
# for i in os.listdir('./CMU_clean'):
# 	pic = cv.imread('./CMU_clean/'+i)
# 	pic = cv.resize(pic,(299,299),fx=0,fy=0) #interpolation = cv2.INTER_CUBIC)
# 	# print(pic.shape)
# 	cv.imwrite('./STAIR_Lab_299/Zsafe/'+i , pic)
# 	c+=1
# 	if c%100==0:
# 		print(c,' videos done')
