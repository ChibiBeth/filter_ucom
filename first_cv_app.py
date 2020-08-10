import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time 
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')

vc = cv2.VideoCapture(0)

if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_eye.xml')

 # Keep the video stream open
while(True):

    # Exit functionality - press any key to exit laptop video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Plot the image from camera with all the face and eye detections marked
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    faces = face_cascade.detectMultiScale(gray, 1.25,6) 
    eyes = eye_cascade.detectMultiScale(gray, 1.25, 6)

    for (x, y, w, h) in faces:
         # Crop the face 
        gray_face_slice = gray[y:y+h, x:x+w]
        color_face_slice = frame[y:y+h, x:x+w]
        # Normalize the gray image for input into the model, so that the values are between [0, 1]
        gray_scale = gray_face_slice / 255
        # Resize the image into 96x96
        original_shape = gray_scale.shape # Keep track of the original size before resizing.
        resized_face = cv2.resize(gray_scale, (96, 96), interpolation = cv2.INTER_AREA)
        resized_face_copy = resized_face.copy()

        resized_face = resized_face.reshape(1, 96, 96, 1) # Resize it further to match expected input into the model

        # Predict
        landmarks = model.predict(resized_face)
        landmarks = landmarks * 48 + 48 # undo the standardization
#       print(landmarks[0][0::2])
#       print(landmarks[0][1::2])
#       print(landmarks[0])
    
        # Plot the landmarks into the color image
        # Plot them in the resized color image of 96x96
        resized_face_color = cv2.resize(color_face_slice, (96, 96), interpolation = cv2.INTER_AREA)
    
        # Resize the 96x96 image to its original size
        # Paste it into the original image
    
        points = []
        for i, co in enumerate(landmarks[0][0::2]):
            points.append((co, landmarks[0][1::2][i]))
        print(len(points))
        for landmark_centre in points:
#           print(landmark_centre)
            cv2.circle(resized_face_color, landmark_centre, 1, (0,255,0), 1)
        
        resized_face_color = cv2.resize(resized_face_color, orig inal_shape, interpolation = cv2.INTER_CUBIC)
    
        frame[y:y+h, x:x+w] = resized_face_color
    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("face detection activated", frame)

    # Read next frame
    time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
    rval, frame = vc.read()
 

