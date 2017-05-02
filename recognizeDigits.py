# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

# load the example image
image = cv2.imread("example.jpg")

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

cv2.imwrite("example2.jpg", edged)
# NOTE: cv2.imwrite(newImageFileName, imageVariable)
# Takes image pixels in imageVariable and writes it to a file newImageFileName

# display image with canny filter applied, DEBUGGING ONLY
# cv2.startWindowThread()
# cv2.namedWindow("preview")
# cv2.imshow("preview", edged)
# if cv2.waitKey():
# 	cv2.destroyAllWindows()

# find contours in the edge map, then sort
# them by their size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# NOTE: cv2.findContours() finds shape outlines in the image
# finds a continuous connected shape
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

# EQUIVALENT TO:
# if imutils.is_cv2():
# 	cnts = cnts[0]
# else:
# 	cnts[1]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if the contour has four vertices, then we have found the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break

# extract the thermostat display, apply a perspective transform to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))

# display the number display (cropped to fit the contour of the display screen outline), DEBUGGING ONLY
# cv2.startWindowThread()
# cv2.namedWindow("preview")
# cv2.imshow("preview", output)
# if cv2.waitKey():
# 	cv2.destroyAllWindows()

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# display the thresholded image (black and white)
# cv2.startWindowThread()
# cv2.namedWindow("preview")
# cv2.imshow("preview", thresh)
# if cv2.waitKey():
# 	cv2.destroyAllWindows()