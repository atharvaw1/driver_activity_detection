import numpy as np
import os
path = '/home/atharva/DAD/AllgoActivityIRData/'
frame_list = []
for root,dir,file in os.walk(path):
    # print('entered')
    if len(dir)==0:
        for i in file:
            if i[-4:] == '.jpg':
                # print(root+'/'+i)
                frame_list.append(root+'/'+i)
# print(frame_list)
print(len(frame_list))
np.save('allgo_ir_index',np.array(frame_list))
