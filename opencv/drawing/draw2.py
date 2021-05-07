# Demo program to draw a rec outline with text on top

import cv2
import numpy as np

win_y = 480
win_x = 640

y = int(win_y/10)
x = int(win_x/10)

pt1 = (x, y)
dx = x + 100
dy = y + 100
pt2 = (dx, dy)

color = (0, 255, 0)

# Create a black image
img = np.zeros((win_y, win_x,3), np.uint8)

# Draw rec
cv2.rectangle(img, pt1, pt2, color, thickness=1)

# Put text in image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(
    img, # image
    'object1', # text that will appear
    pt1, # BL corner of textbox
    font, # font specified
    1, #
    color,
    2, # line type
    cv2.LINE_AA
)

# Displaying the image
cv2.imshow("Drawing", img)
print('Press any key to quit.')
cv2.waitKey()
