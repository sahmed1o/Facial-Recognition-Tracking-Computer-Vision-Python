import cv2
import os
import numpy as np


#classifiers to capture eyes and face
face_cascade = cv2.CascadeClassifier('haarcascades\\haarcascade_frontalface_default.xml');

cam=cv2.VideoCapture(0); #video capture id, 0 will be webcam id

#Capture the face
while True:
    ret,img = cam.read(); #return status variable and the captured image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #grayscale image for detection filtering of eyes
    faces = face_cascade.detectMultiScale(gray,1.3,5); #detect all the faces in current frame and return cooorinate of face

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle on coordinates of face captured           
        cv2.waitKey(100) # short pause of r100 seconds

    cv2.imshow("FaceDetectionOpenCV", img);
    key = cv2.waitKey(1)
    if key == 27: #esc key is pressed
        break

    
cam.release()
cv2.destroyAllWindows()
