
# import the necessary packages
from shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]

#kernel = np.ones((3,3),np.uint8) 
#thresh = cv2.erode(thresh,kernel,iterations = 3)


#cv2.imshow("binary", thresh)
#cv2.waitKey(0)

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

# loop over the contours
index = 1
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 0, 0), 2)

	print(c)
	row = c.shape[0]
	col = c.shape[2]
	ptArray = np.resize(c, (row, col))
	print(ptArray)
	print(ptArray.shape)
	area = 0
	perimeter = 0
	perimeter = cv2.arcLength(c, True)
	area = cv2.contourArea(c)
	print(str(index) + " - " + str(shape))
	print("area : " + str(area))
	print("perimeter : " + str(perimeter))
	print("----------------------")
	index = index + 1
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
    
