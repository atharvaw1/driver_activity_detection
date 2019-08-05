import numpy as np

allgo = np.load('allgo_x_224.npy',allow_pickle=True)
serre = np.load('serre_x_224.npy',allow_pickle=True)
stair = np.load('stair_x_224.npy',allow_pickle=True)
stair_z  =np.load('stair_x_224_safe.npy',allow_pickle=True)

allgo_d = allgo[:80644]
allgo_t = allgo[80644:]
# print(allgo_d,allgo_d.shape)
# print(allgo_t ,allgo_t.shape)
print(allgo_d.shape)
print(allgo_t.shape)
np.random.shuffle(allgo_d)
np.random.shuffle(allgo_t)
allgo_d = allgo_d[:-60000]
allgo_t = allgo_t[:-50000]
print(allgo_d.shape)
print(allgo_t.shape)

serre_d = serre[:10776]
serre_e = serre[10776:10776+6167]
serre_z = serre[10776+6167:10776+6167+35744]
serre_t = serre[10776+6167+35744:10776+6167+35744+1795]
serre_s = serre[10776+6167+35744+1795:]
# print(serre_d , serre_d.shape)
# print(serre_e , serre_e.shape)
# print(serre_z , serre_z.shape)
# print(serre_t , serre_t.shape)
# print(serre_s , serre_s.shape)


stair_d = stair[:53662]
stair_e = stair[53662:53662+49743]
stair_s = stair[53662+49743:53662+49743+42206]
stair_t = stair[53662+49743+42206:]
# print(stair_d,stair_d.shape)
# print(stair_e,stair_e.shape)
# print(stair_s,stair_s.shape)
# print(stair_t,stair_t.shape)
print(stair_d.shape)
print(stair_t.shape)
print(stair_z.shape)
np.random.shuffle(stair_d)
np.random.shuffle(stair_t)
np.random.shuffle(stair_z)
stair_d = stair_d[:-20000]
stair_t = stair_t[:-60000]
stair_z = stair_z[:-70000]
print(stair_d.shape)
print(stair_t.shape)
print(stair_z.shape)


combine = np.concatenate((allgo_d,allgo_t,stair_d,stair_e,stair_s,stair_t,stair_z,serre_e,serre_d,serre_s,serre_t,serre_z),axis=0)
print(combine.shape)
print(combine)
np.save('dataset_combined',combine)
