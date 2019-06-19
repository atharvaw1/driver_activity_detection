import numpy as np

for i in range(1):

	d = np.load('bottleneck_files/test.npy',allow_pickle=True)
	for j in d:
		np.save('bottleneck_data_sorted2/'+j[0][16:]+j[1] , j[2:])

	print('file:',i)
