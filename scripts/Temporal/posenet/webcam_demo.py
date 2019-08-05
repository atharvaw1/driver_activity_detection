import tensorflow as tf
import cv2
import time
import argparse
import posenet
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq


def isInsideC(circle_x, circle_y, rad, x, y): 
      
    # Compare radius of circle 
    # with distance of its center 
    # from given point 
    if ((x - circle_x) * (x - circle_x) + 
        (y - circle_y) * (y - circle_y) <= rad * rad): 
        return True
    else: 
        return False
 



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

def dist_from_line(x,y,ps1,ps2):
    points = [ps1,ps2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    # print(A,y_coords)
    m, c = lstsq(A, y_coords)[0]
    # print("Line Solution is y = {m}x + {c}".format(m=m,c=c))

    d = (abs(m*x - y + c))/((m*m+1)**(1/2))
    return d

def dist_from_pt(x,y,pt):
    x2 =pt[0]
    y2 = pt[1]
    # if x2 < 0 or y2 <0 or y2>230 or x2>230:
    #     # print('jere')
    #     return 999999
    d = ((x-x2)**2 + (y-y2)**2)**0.5
    return d





parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=224)#1280)
parser.add_argument('--cam_height', type=int, default=224)#720)
parser.add_argument('--scale_factor', type=float, default=1)#0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():

    with tf.Session() as sess:

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        hasFrame, frame = cap.read()
        # print(frame.shape)
        start = time.time()
        frame_count = 0
        prvs = cv2.resize(frame,(224,224))
        prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros((224,224,3),dtype=np.uint8)
        hsv[...,1] = 255 #intensity
        c = 0




        while True:

            c += 1
            flag = 0
            t = time.time()
            hasFrame, frame = cap.read()
            if not hasFrame:
                break

            next = cv2.resize(frame,(224,224))
            # print(next.shape)
            next = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
            prvs = cv2.medianBlur(prvs,5)
            next = cv2.medianBlur(next,5)
            # print(prvs.shape,next.shape)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5,3,7,4,7,5, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mag = (mag>1.4)*mag
            
            hsv[...,0] = ang*180/np.pi/2 #hue, colour
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) #brightness
            
            up_mask = make_mask(hsv[...,0],125,150) #purple
            down_mask = make_mask(hsv[...,0],35,75) #green
            left_mask = make_mask(hsv[...,0],165,179) |  make_mask(hsv[...,0],0,30)#red
            right_mask = make_mask(hsv[...,0],75,105) #blue

            #label machine
            # up_mask = make_mask(hsv[...,0],125,150) #purple
            # down_mask = make_mask(hsv[...,0],35,75) #green
            # left_mask = make_mask(hsv[...,0],165,179) |  make_mask(hsv[...,0],1,20)#red
            # right_mask = make_mask(hsv[...,0],80,100) #blue


            
            hsv_up = apply_mask(hsv,up_mask)
            hsv_down = apply_mask(hsv,down_mask)
            hsv_left = apply_mask(hsv,left_mask)
            hsv_right = apply_mask(hsv,right_mask)





            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)

            keypoint_coords *= output_scale
            #for i in range(len(pose_scores)):


            for pts in keypoint_coords:
                dist1 = [0]
                dist2 = [0]
                if int(pts[9,0]) >0 and int(pts[9,0]) < 140 and int(pts[9,1]) >0 and int(pts[9,1]) < 140:
                    cv2.circle(display_image,(int(pts[9,1]),int(pts[9,0])-20), 20, (0,255,0), 2)
                    cv2.circle(display_image,(int(pts[9,1]),int(pts[9,0])+30), 20, (0,255,0), 2)
                if int(pts[10,0]) >0 and int(pts[10,0]) < 140 and int(pts[10,1]) >0 and int(pts[10,1]) < 140:
                    cv2.circle(display_image,(int(pts[10,1]),int(pts[10,0])-20), 20, (0,255,0), 2)
                    cv2.circle(display_image,(int(pts[10,1]),int(pts[10,0])+30), 20, (0,255,0), 2)
                valid_pts = 0
                for i in range(len(mag)):
                    for j in range(len(mag[0])):
                        if mag[j,i]>0:

                            if (isInsideC(int(pts[9,1]),int(pts[9,0])-20,20,j,i) or isInsideC(int(pts[9,1]),int(pts[9,0])+30,20,j,i)) and (int(pts[9,0]) >0 and int(pts[9,0]) < 140 and int(pts[9,1]) >0 and int(pts[9,1]) < 140):
                                valid_pts+=1
                            if (isInsideC(int(pts[10,1]),int(pts[10,0])-20,20,j,i) or isInsideC(int(pts[10,1]),int(pts[10,0])+30,20,j,i)) and (int(pts[10,0]) >0 and int(pts[10,0]) < 140 and int(pts[10,1]) >0 and int(pts[10,1]) < 140):
                                valid_pts+=1                                                             
                            # dist1.append(dist_from_line(j,i,pts[7,:],pts[9,:])) # left hand
                            # dist2.append(dist_from_line(j,i,pts[8,:],pts[10,:]))# right hand
                            # dist1.append(dist_from_pt(j,i,pts[9,:])) # left wrist
                            # dist2.append(dist_from_pt(j,i,pts[10,:]))  # right wrist
                            # print(pts[9,:])
                            # print(pts[10,:])
            # print(valid_pts)


            # print(np.mean(dist1),np.mean(dist2))
            thresh = 140
            up_thresh = 34
            down_thresh = 14
            left_thresh = 24
            right_thresh = 28


            # if (np.mean(dist1) < thresh or np.mean(dist2)<thresh) and (np.mean(dist1) >0 and np.mean(dist2)>0):
            # if True:
            if valid_pts > 100:
                #original
                # print('please print')
                # print(np.mean(hsv_right[...,0]))
                if np.mean(hsv_up[...,0])>up_thresh  and np.mean(mag)>0.07:
                    # print(np.mean(hsv_up[...,0]))
                    print('UP',c)
                    flag = 1

                if np.mean(hsv_down[...,0])>down_thresh  and np.mean(mag)>0.07:
                    # print(np.mean(hsv_down[...,0]))
                    print('DOWN',c)
                    flag = 1

                if np.mean(hsv_left[...,0])>left_thresh and np.mean(mag)>0.07:
                    # print(np.mean(hsv_left[...,0]))
                    print('LEFT',c)
                    flag = 1

                if np.mean(hsv_right[...,0])>right_thresh  and np.mean(mag)>0.07:
                    # print(np.mean(hsv_right[...,0]))
                    print('RIGHT',c)
                    flag = 1

                #modified
                # if np.mean(hsv_up[...,0])>38 and np.mean(mag)>0.08:
                #     print('UP',np.mean(hsv_up[...,0]))
                #     flag = 1

                # if np.mean(hsv_down[...,0])>16.5 and np.mean(mag)>0.08:
                #     print('DOWN',np.mean(hsv_down[...,0]))
                #     flag = 1

                # if np.mean(hsv_left[...,0])>24 and np.mean(mag)>0.08:
                #     print('LEFT',c)
                #     flag = 1

                # if np.mean(hsv_right[...,0])>28 and np.mean(mag)>0.08:
                #     print('RIGHT',c)
                #     flag = 1


            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            overlay_image = cv2.resize(overlay_image,(224,224))



            bgru = cv2.cvtColor(hsv_up,cv2.COLOR_HSV2BGR)
            bgrd = cv2.cvtColor(hsv_down,cv2.COLOR_HSV2BGR)
            bgrl = cv2.cvtColor(hsv_left,cv2.COLOR_HSV2BGR)
            bgrr = cv2.cvtColor(hsv_right,cv2.COLOR_HSV2BGR)
            # bgr = cv2.medianBlur(bgr,5)
            
            cv2.imshow('flow_up',bgru)
            cv2.imshow('flow_down',bgrd)
            cv2.imshow('flow_left',bgrl)
            cv2.imshow('flow_right',bgrr)
            cv2.imshow('posenet', overlay_image)
            prvs = next
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
