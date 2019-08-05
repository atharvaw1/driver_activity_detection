import cv2 as cv
import time
import numpy as np

# cam = cv2.VideoCapture(0)
# # z = time.time()
# i = 0
# frame_list = []
# while i<150:
#     ret_val, img = cam.read()
#     cv2.imshow('my webcam', img)
#     if cv2.waitKey(1) == 27: 
#         break  # esc to quit
#     i += 1
#     frame_list.append(img)

# np.save('gowda_optical',frame_list)
d = np.load('gowda_optical_smoking.npy',allow_pickle=True)
for i in d:
	cv.imshow('frame',i)
	# print(d[0].shape)
	if cv.waitKey(20) == 27:
		break
cv.destroyAllWindows()