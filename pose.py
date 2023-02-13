import math 
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from time import time
# Initalizing the mediapipe pose class
mp_pose=mp.solutions.pose

pose=mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2,smooth_landmarks=False)
mp_drawing=mp.solutions.drawing_utils

pose_video=mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.3,model_complexity=2)
def detectPose(image,pose,display=True):
    
    output_image=image.copy()
    imageRGB=image[:,:,::-1]
    results=pose.process(imageRGB)
    height,width,_=image.shape
    landmarks=[]
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image,landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x*width),int(landmark.y*height),(landmark.z*width)))
            
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121)
        plt.imshow(image[:,:,::-1])
        plt.title("original image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:,:,::-1])
        plt.title("output image")
        plt.axis('off')
    else:
        return output_image,landmarks


video=cv2.VideoCapture(0)

time1=0 
while video.isOpened():
    ok,frame=video.read()
    if not ok :
        break 
    frame =cv2.flip(frame,1)
    frame_height,frame_width,_ =frame.shape
    frame=cv2.resize(frame,(int(frame_width*(640/frame_height)),640))
    frame,_=detectPose(frame,pose_video,display=False)
    time2=time()
    if (time2-time1)>0:
        frames_per_second=1.0/(time2-time1)
        
        cv2.putText(frame,'FPS:{}'.format(int(frames_per_second)),(10,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
    time1=time2
    cv2.imshow('pose detection',frame)
    k=cv2.waitKey(1) & 0xFF
    
    if(k==27):
        break
video.release()
cv2.destroyAllWindows()
