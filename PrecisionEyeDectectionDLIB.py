import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

#refer to the 68-face-landmarks-labeled-by-dlib-software-automatically.png to understand why certain coordinates are used to find certain parts of the face

detector = dlib.get_frontal_face_detector()  #front face classifier
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #assigned coordinates of the face by DLIB

while True:
    ret, frame = cap.read() #return status variable and the captured image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale image for detection filtering of eyes

    faces = detector(gray) #array of all faces
    for face in faces:
        x, y  = face.left(), face.top() # Top Left coordinates of face in window
        x1, y1 = face.right(), face.bottom() # Bottom right coordinates of face in windows
        cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2) # form a rectangle based on previous two coordinates

        poslandmarkpoints = predictor(gray, face)
        
        #Precision Tracking Of Eyes
        #Left Eye Tracking
        leftEyeTrack = np.array([(poslandmarkpoints.part(36).x,poslandmarkpoints.part(36).y),
                                 (poslandmarkpoints.part(37).x,poslandmarkpoints.part(37).y),
                                 (poslandmarkpoints.part(38).x,poslandmarkpoints.part(38).y),
                                 (poslandmarkpoints.part(39).x,poslandmarkpoints.part(39).y),
                                 (poslandmarkpoints.part(40).x,poslandmarkpoints.part(40).y),
                                 (poslandmarkpoints.part(41).x,poslandmarkpoints.part(41).y)
                                 ],
                                np.int32)
        cv2.polylines(frame, [leftEyeTrack], True, (0,255,0), 2)

        
        #Right Eye Tracking
        rightEyeTrack = np.array([(poslandmarkpoints.part(42).x,poslandmarkpoints.part(42).y),
                                 (poslandmarkpoints.part(43).x,poslandmarkpoints.part(43).y),
                                 (poslandmarkpoints.part(44).x,poslandmarkpoints.part(44).y),
                                 (poslandmarkpoints.part(45).x,poslandmarkpoints.part(45).y),
                                 (poslandmarkpoints.part(46).x,poslandmarkpoints.part(46).y),
                                 (poslandmarkpoints.part(47).x,poslandmarkpoints.part(47).y)
                                 ],
                                np.int32)
        cv2.polylines(frame, [rightEyeTrack], True, (0,255,0), 2)
                
    cv2.imshow("Pupil Detection With DLIB", frame)
    key = cv2.waitKey(1)
    if key == 27: #esc key is pressed
        break


cap.release()
cv2.destroyAllWindows()
