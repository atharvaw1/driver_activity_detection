import numpy as np
import os
for i in range(9):
	frame = 0
	d = np.load('bottleneck_files/{0}.npy'.format(36+i),allow_pickle=True)
	for j in d:
		# print('bottleneck_data_sorted/'+j[0][16:], j[2:])
		np.save('bottleneck_data_sorted2/'+j[0][21:-4]+j[1], j[2:])
		# print('bottleneck_data_sorted2/'+j[0][21:-4]+j[1])
		# break

		frame+=1
		if frame%1000==0:
			print(frame)
	os.remove('bottleneck_files/{0}.npy'.format(36+i))


	# break
	print('file:',i)
