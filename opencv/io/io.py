# Draw with mouse demo

import cv2
import numpy as np



# Mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 25, (0,255,255),-1)



# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
window_name = 'Draw with Mouse!'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_circle)

print('Double click to draw a circle!')
print("Press 'esc' to quit.")
while(1):
    cv2.imshow(window_name,img)
    # Press esc key to exit from window
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
