# A more advanced mouse io demo
import cv2
import numpy as np
import time

# Config
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1


# Mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                x=0
                #cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 255), 1)
                #time.sleep(.01)
                #cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 255), 1)
            else:
                cv2.circle(img,(x,y),5,(0,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0, 0, 255),thickness=1)
        else:
            cv2.circle(img,(x,y),5,(0,255,255),-1)



w_name = 'draw!'
img = np.zeros((512,512,3), np.uint8) # start with a black screen
cv2.namedWindow(w_name)

# Track mouse movement (specify actions in callback) in specified window
cv2.setMouseCallback(w_name, draw_circle)

while(1):
    cv2.imshow(w_name, img)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()
