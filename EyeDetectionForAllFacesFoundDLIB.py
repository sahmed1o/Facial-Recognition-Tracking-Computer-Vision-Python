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
        cv2.putText(frame, 'Coord('+str(x)+','+str(y)+')', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2) #show position of face
        
        poslandmarkpoints = predictor(gray, face)
        
        # LEFT EYE TRACKING WITH DLIB
        # Using the DLIB landmarks diagram, coordinates 38 will be used for length of box
        # Width
        #(landmarks.part(38).x, landmarks.part(38).y)   coordinate for left most eye to determine width of eye rectangle
        
        # Using the DLIB landmarks diagram, coordinates 41 will be used for height of box
        # Height
        #(landmarks.part(41).x, landmarks.part(41).y)   first coordinate for left most eye to determine width of eye rectangle

        #x1,y1 ------
        #|          |
        #|          |
        #|          |
        #--------x2,y2
        
        left_eye1x = poslandmarkpoints.part(38).x
        left_eye1y = poslandmarkpoints.part(38).y
        left_eye2x = poslandmarkpoints.part(41).x
        left_eye2y = poslandmarkpoints.part(41).y
        
        cv2.rectangle(frame, (left_eye1x-40,left_eye1y-20), (left_eye2x+40,left_eye2y+20), (0,255,0), 2)
        cv2.putText(frame, 'Coord('+str(left_eye1x)+','+str(left_eye1y)+')', (left_eye1x-30, left_eye1y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2) #show position of face
        
        
        # RIGHT EYE TRACKING WITH DLIB
        # Using the DLIB landmarks diagram, coordinates 44 will be used for length of box
        # Width
        #(landmarks.part(44).x, landmarks.part(44).y)   coordinate for left most eye to determine width of eye rectangle
        
        # Using the DLIB landmarks diagram, coordinates 47 will be used for height of box
        # Height
        #(landmarks.part(47).x, landmarks.part(47).y)   first coordinate for left most eye to determine width of eye rectangle

        
        #x1,y1 ------
        #|          |
        #|          |
        #|          |
        #--------x2,y2
        
                
        right_eye1x = poslandmarkpoints.part(44).x
        right_eye1y = poslandmarkpoints.part(44).y
        right_eye2x = poslandmarkpoints.part(47).x
        right_eye2y = poslandmarkpoints.part(47).y
        
        cv2.rectangle(frame, (right_eye1x-40,right_eye1y-20), (right_eye2x+40,right_eye2y+20), (0,255,0), 2)
        cv2.putText(frame, 'Coord('+str(right_eye1x)+','+str(right_eye1y)+')', (right_eye1x-60, right_eye1y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2) #show position of face
        
    cv2.imshow("Pupil Detection With DLIB", frame)
    key = cv2.waitKey(1)
    if key == 27: #esc key is pressed
        break


cap.release()
cv2.destroyAllWindows()
