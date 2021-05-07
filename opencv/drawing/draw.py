'''
In all the above functions, you will see some common arguments as given below:

    - img : The image where you want to draw the shapes
    - color : Color of the shape. for BGR, pass it as a tuple, eg: (255,0,0) for blue. For grayscale, just pass the scalar value.
    - thickness : Thickness of the line or circle etc. If -1 is passed for closed figures like circles, it will fill the shape. default thickness = 1
    - lineType : Type of line, whether 8-connected, anti-aliased line etc. By default, it is 8-connected. cv2.LINE_AA gives anti-aliased line which looks great for curves.
'''

import cv2
import numpy as np

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)

# Draw polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))

# Put text in image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Text here',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

# Displaying the image
cv2.imshow("Drawing", img)
print('Press any key to quit.')
cv2.waitKey()
