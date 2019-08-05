import numpy as np
import re

path = '../atharva_old/Dataset_temporal/Action/'
d = 0

# d = np.load('stair_x_224.npy',allow_pickle=True)
# video_list = {}

# # low = 24
# # high = 34

# def get_string(path,mobj):
# 	return path[mobj.start():mobj.end()]
# vid = 0
# old = -1000
# while vid < d.shape[0]:
# 	c = 0
# 	k = d[vid][0]
# 	vid_no = re.search("a[0-9]{3}-[0-9]{4}C",k)
# 	# print(desc.span(),k[desc.start():desc.end()])
# 	# print(d.shape)
# 	for i in d:
# 		search_no = re.search("a[0-9]{3}-[0-9]{4}C",i[0])
# 		if get_string(i[0],search_no) == get_string(k,vid_no):
# 			# print(i)
# 			c += 1
# 	# print(k,c)	
# 	vid += c
# 	video_list[k] = c
# 	if (vid-old)>1000:
# 		print('Processed: {0}  video_list: {1}'.format(vid,len(video_list))) 
# 		old = vid
# print('Saving dictionary')
# np.save('stair_name_dict',video_list)
# print('Saved ', len(video_list), ' videos')

for f in os.listdir(path+str(d)):
	d += 1
	