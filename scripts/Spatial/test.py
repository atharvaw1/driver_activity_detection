import numpy as np
import os

d = np.load('dataset_combined.npy',allow_pickle=True)
print(d.shape)






# f = np.load('dataset_x_spatial.npy',allow_pickle=True)
# c=0
# frame = 0
# print(len(f))
# del_list = []
# for i,video in enumerate(f[55000:]):
# 	if video[1] == '-1':
# 		path = 'bottleneck_data_sorted/'+video[0][16:]+'.npy'
# 	else:
# 		path = 'bottleneck_data_sorted/'+video[0][16:]+video[1]+'.npy'
# 	frame+=1
# 	if frame%1000==0:
# 		print('Frame:',frame+55000)
# 	try:
# 		d = np.load(path,allow_pickle=True)
# 		del d
# 	except:
# 		c+=1
# 		del_list.append(i+55000)
# 		# print(path)
# 		# print(c)
# print(f.shape)
# f = np.delete(f,del_list,axis=0)
# print(f.shape)
# np.save('dataset_x_improved',f)
