import numpy as np
import cv2


d = np.load('stairlab_dict.npy',allow_pickle=True)
d = d.item()
for i in d:
	
	img_list=[]


	for j in range(4,-1,-1):
		# print(i)
		# print(i[i.rfind('_')+1:i.rfind('.')])
		path = '../atharva_old/'+i[2:i.rfind('_')+1] + str(int(i[i.rfind('_')+1:i.rfind('.')])-j)+'.png'
		img = cv2.imread(path)

		img_list.append(img)
		

	for j in range(5):
		# print(i)
		# print(i[i.rfind('_')+1:i.rfind('.')])
		path = '../atharva_old/'+i[2:i.rfind('_')+1] + str(int(i[i.rfind('_')+1:i.rfind('.')])+j)+'.png'
		img = cv2.imread(path)

		img_list.append(img)	



	for img in img_list:
		try:
			print(d[i])
			cv2.imshow('frame',img)

		except:
			print(path)
		if cv2.waitKey(200) & 0xFF == ord('q'):
			exit()