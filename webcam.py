
#pip3 install opencv-python
#pip3 install opencv-contrib-python

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

if cap.read() == False:
    cap.open()

if not cap.isOpened():
    print('Cannot open camera')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #import pdb; pdb.set_trace()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
