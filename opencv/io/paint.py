#*********************************
# Use trackbars to adjust colors when painting
#*********************************
import cv2
import numpy as np

#=================================
# Functions
#=================================
def convert(list):
    return tuple(list)

def nothing(x):
    pass

# Mouse event callback for drawing a circle
def draw(event, x, y, flags, param):
    global ix, iy, drawing
    radius = 4 # radius of circle
    rgb = convert(color) # get color tuple
    
    # Handle different mouse actions
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            #print('color=(r={}, g={}, b={})'.format(r, g, b))
            cv2.circle(img, (x, y), radius, rgb, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), radius, rgb, -1)

#=================================
# Start with program
#=================================
# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)

# Color tuple
color = [0, 0, 0]

# Other drawing-related vars
drawing = False # true if mouse is pressed
ix, iy = -1, -1

# Name window with wn
wn = 'image'
cv2.namedWindow(wn)

# create trackbars for color change
cv2.createTrackbar('R',wn,0,255,nothing)
cv2.createTrackbar('G',wn,0,255,nothing)
cv2.createTrackbar('B',wn,0,255,nothing)

#=================================
# Main loop: keep drawing or exit with ESC key
#=================================
while(1):
    cv2.imshow(wn,img)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # esc
        break

    # Use trackbars to adjust bgr values
    r = cv2.getTrackbarPos('R',wn)
    g = cv2.getTrackbarPos('G',wn)
    b = cv2.getTrackbarPos('B',wn)
    color[0] = b
    color[1] = g
    color[2] = r
    
    # Track mouse movement
    cv2.setMouseCallback(wn, draw)

cv2.destroyAllWindows()
