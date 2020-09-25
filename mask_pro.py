import numpy as np
import cv2
import random

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")
mouth_cascade=cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
upper_cascade=cv2.CascadeClassifier("haarcascade_upperbody.xml")

bw=80

font=cv2.FONT_HERSHEY_SIMPLEX
org=(30,30)
weared_mask_font=(255,255,255)
not_wear_mask_font=(0,0,255)
thickness=2
font_scale=1
wear_mask="Thank You for wearing a Mask"
not_wear_mask="Please wear a Mask to defeat Corona"

cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    img=cv2.flip(img,1)

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, black_white)=cv2.threshold(gray, bw, 255, cv2.THRESH_BINARY)
    cv2.imshow('black_white', black_white)

    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    faces_bw=face_cascade.detectMultiScale(black_white, scaleFactor=1.1, minNeighbors=4)

    if len(faces)==0 and len(faces_bw)==0:
        cv2.putText(img, "No faces found...", org, font, font_scale, weared_mask_font, thickness, cv2.LINE_AA)
    elif len(faces)==0 and len(faces_bw)==1:
        cv2.putText(img, wear_mask, org, font, font_scale, weared_mask_font, thickness, cv2.LINE_AA)
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255),2)
            roi_gray=gray[y:y+h, x:x+w]
            roi_colour=img[y:y+h, x:x+w]
            mouth_rects=mouth_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            if(len(mouth_rects) == 0):
                cv2.putText(img, wear_mask, org, font, font_scale, weared_mask_font, thickness, cv2.LINE_AA)
            else:
                for (mx, my, mw, mh) in mouth_rects:

                    if(y < my < y + h):
                        cv2.putText(img, not_wear_mask, org, font, font_scale, not_wear_mask_font, thickness, cv2.LINE_AA)
                        cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                        break

    cv2.imshow("Mask Detection", img)
    k=cv2.waitKey(30) & 0xff
    if k==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
