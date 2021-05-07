# OpenCV video tutorial

import cv2
import numpy as np

print('Note: press \'q\' to quit.')

#---------------------------------
# Capture video from a camera
#---------------------------------

cap = cv2.VideoCapture(0)

# Set screen resolution
cap.set(3, 640)
cap.set(4, 480)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #res = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # grayscale
    #res = cv2.bitwise_not(frame)
    #res = cv2.Canny(frame, 100, 200) # apply Canny edge detect

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
#'''


#---------------------------------
# Play a video from file
#---------------------------------
'''
cap = cv2.VideoCapture('metro.mov')
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#'''


#---------------------------------
# Save modified video
# Define the codec and create VideoWriter object
#---------------------------------
'''
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        frame = cv2.flip(frame,0)
        out.write(frame) # write the flipped frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# Release everything if job is finished
cap.release()
out.release()
#'''

cv2.destroyAllWindows()

