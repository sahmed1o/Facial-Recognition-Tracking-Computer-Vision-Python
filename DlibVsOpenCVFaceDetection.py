import cv2
import os
import numpy as np
import dlib

#classifiers to capture eyes and face
face_cascade = cv2.CascadeClassifier('haarcascades\\haarcascade_frontalface_default.xml');

cam=cv2.VideoCapture(0); #video capture id, 0 will be webcam id

detector = dlib.get_frontal_face_detector() #front face classifier

#Capture the face
while True:
    ret,img = cam.read(); #return status variable and the captured image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #grayscale image for detection filtering of eyes
    faces1 = face_cascade.detectMultiScale(gray,1.3,5); #detect all the faces in current frame and return cooorinate of face
    faces2 = detector(gray) #array of all faces
    
    for(x,y,w,h) in faces1:

        #OpenCV face detection, drawing blue rectangle
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #draw rectangle on coordinates of face captured 
        cv2.putText(img, 'OPenCV Face Detection', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2) #show text
        
    for face in faces2:
        #Dlib face detection, drawing green rectangle
        x1, y1  = face.left(), face.top() # Top Left coordinates of face in window
        x2, y2 = face.right(), face.bottom() # Bottom right coordinates of face in windows
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2) # form a rectangle based on previous two coordinates
        cv2.putText(img, 'DLIB Face Detection', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2) #show text
        
    cv2.imshow("DlibVsOpenCVFaceDetection", img);
    key = cv2.waitKey(1)
    if key == 27: #esc key is pressed
        break

    
cam.release()
cv2.destroyAllWindows()
