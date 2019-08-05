import numpy as np
import cv2
cap = cv2.VideoCapture(0)
# cap = cv.VideoCapture("./Test_videos/drink1.avi")
# cap = cv.VideoCapture("./Test_videos/drink2.avi")
# cap = cv.VideoCapture("./Test_videos/smoke1.avi")
# cap = cv.VideoCapture("./Test_videos/smoke2.avi")
# cap = cv.VideoCapture("./Test_videos/phone1.avi")
# cap = cv.VideoCapture("./Test_videos/phone2.avi")
# cap = cv.VideoCapture("./Test_videos/safe1.avi")
# cap = cv.VideoCapture("./Test_videos/safe2.avi")
# cap = cv.VideoCapture("./Test_videos/eat1.avi")
# cap = cv.VideoCapture("./Test_videos/eat2.avi")


ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros((224,224,3),dtype=np.uint8)
prvs = cv2.resize(prvs,(224,224))
hsv[...,1] = 255 #intensity
c = 0

def make_mask(arr,lower,upper):
    mask1 = arr>lower
    mask2 = arr<upper
    mask = mask1 & mask2
    return mask

def apply_mask(hsv,mask):
    hsv_copy = hsv.copy()
    hsv_copy[...,0] = mask * hsv_copy[...,0]
    hsv_copy[...,2] = mask * hsv_copy[...,2]
    return hsv_copy

while(ret):
    c += 1
    flag = 0
    ret, frame2 = cap.read()

    if not ret:
        break

    next = cv2.resize(frame2,(224,224))
    next = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
    prvs = cv2.medianBlur(prvs,5)
    next = cv2.medianBlur(next,5)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5,3,7,4,7,5, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag = (mag>1.4)*mag
    
    hsv[...,0] = ang*180/np.pi/2 #hue, colour
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) #brightness
    
    up_mask = make_mask(hsv[...,0],125,150) #purple
    down_mask = make_mask(hsv[...,0],35,75) #green
    left_mask = make_mask(hsv[...,0],165,179) |  make_mask(hsv[...,0],1,20)#red
    right_mask = make_mask(hsv[...,0],80,100) #blue

    
    hsv_up = apply_mask(hsv,up_mask)
    hsv_down = apply_mask(hsv,down_mask)
    hsv_left = apply_mask(hsv,left_mask)
    hsv_right = apply_mask(hsv,right_mask)
    
    # hsv_test = hsv.copy()
    # for i in range(len(mag)):
    #     for j in range(len(mag[0])):
    #         if mag[i,j] > 28:
    #         # print(hsv_test.shape)
    #             print('y:',i,'x:',j)


    if np.mean(hsv_up[...,0])>38 and np.mean(mag)>0.08:
        print('UP',np.mean(hsv_up[...,0]))
        flag = 1

    if np.mean(hsv_down[...,0])>16.5 and np.mean(mag)>0.08:
        print('DOWN',np.mean(hsv_down[...,0]))
        flag = 1

    if np.mean(hsv_left[...,0])>24 and np.mean(mag)>0.08:
        print('LEFT',c)
        flag = 1

    if np.mean(hsv_right[...,0])>28 and np.mean(mag)>0.08:
        print('RIGHT',c)
        flag = 1

        # if flag==0:
        #     print('NO MOVEMENT',c)

    
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    bgr = cv2.medianBlur(bgr,5)
    
    cv2.imshow('frame2',bgr)
    cv2.imshow('original',prvs)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next
cap.release()
cv2.destroyAllWindows()
