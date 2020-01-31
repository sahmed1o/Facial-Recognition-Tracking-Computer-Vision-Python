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
        cv2.rectangle(frame, (x,y), (x1,y1), (0, 255,0), 2) # form a rectangle based on previous two coordinates

        
        poslandmarkpoints = predictor(gray, face) #grab all landmark coordinates
        for p in range(0, 68):
            x = poslandmarkpoints.part(p).x #x coordinate of landmark at p value
            y = poslandmarkpoints.part(p).y #y coordinate of landmark at p value
            cv2.circle(frame, (x,y), 3, (0, 255, 0), 2) #form a dot on each landmark position, DLIB tracks up to 68
        
    cv2.imshow("Pupil Detection With DLIB", frame)
    key = cv2.waitKey(1)
    if key == 27: #esc key is pressed
        break


cap.release()
cv2.destroyAllWindows()
