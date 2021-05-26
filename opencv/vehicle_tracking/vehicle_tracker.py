# Vehicle Detection using Cascade Classifier object
# NOTE: if camera is moving this trained classifier performs poorly
import cv2
import numpy as np
import sys

# Read in file name
filename = sys.argv[1]
cap = cv2.VideoCapture(filename)
car_cascade = cv2.CascadeClassifier('cars_given.xml') # take in pre-trained classifier (not from me)

while True:
	# Read frame in
	ret, frame = cap.read()

	if ret is True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cars = car_cascade.detectMultiScale(gray, 1.1, 1) # detect cars of different sizes

		# Draw bounding box around cars
		for (x,y,w,h) in cars:
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

		cv2.imshow('Video Feed', frame)

		# Press 'q' to quit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		cv2.waitKey(10)

cv2.destroyAllWindows()
