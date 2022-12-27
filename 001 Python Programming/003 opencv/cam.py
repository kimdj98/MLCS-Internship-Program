# -*- coding: utf-8 -*-
"""cam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17pVpqhfqtaUrx66nC2BC2vp-6u-0-eTs

## Basic Python Programming 003 opencv

#### image preprocessing
"""

import cv2
import imutils
from imutils import face_utils

image = cv2.imread("person.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("Original", image)

height, width, channel = image.shape # get height, width
ratio =  600.0 / height
image = imutils.resize(image, height=600, width=int(width*ratio))
# cv2.imshow("Shrinked", image)

image = imutils.rotate_bound(image, 90)
# cv2.imshow("rotate image", image)

cv2.waitKey(0)
# You can close the window by pressing any key.
cv2.destroyAllWindows()

"""#### Detect face and put sunglasses on person.jpg"""

import cv2
import dlib
import imutils
from imutils import face_utils

import numpy as np

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess_sunglasses_image(image, output_width):
    height, width, channel = image.shape # get height, width
    ratio = output_width / width
    image = imutils.resize(image, height=int(height*ratio), width=output_width)
    # image = imutils.rotate_bound(image,90)
    return image

while True:
    # load the input image and convert it to grayscale
    image = image
    gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects): # rects: Rectangle instance(collection of rectangles)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect) # shape: coordinates of facial landmarks
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        count = 0
        for (x, y) in shape:
            if count in {36}:
                left_eye_coordinate = (x, y)
            # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            count += 1

    l = rect.left()
    r = rect.right()
    t = rect.top()
    b = rect.bottom()

    # draw rectangle to check face detection
    # cv2.rectangle(image, (l,t),(r,b),(0,255,0), 1)

    # make cropped image
    cropped_image = image[t:b,l:r]

    # read sunglasses.png
    # alpha channel is used to determine the transparancy of image
    sunglasses_image_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED) # get 3 dim data rgba (r,g,b,a:transparancy)

    # prerpocess sunglasses_image
    sunglasses_image_png = preprocess_sunglasses_image(sunglasses_image_png, output_width = r-l)

    # read sunglass dimensions
    height, width, channel = sunglasses_image_png.shape
    y = left_eye_coordinate[1]-t
    # print(cropped_image.shape)
    # print(sunglasses_image_png.shape)
    # print(y)
    for i in range(height):
        for j in range(width):
            if sunglasses_image_png[i, j, 3] == 0:
                pass
            else:
                cropped_image[y-29//2+3+i,j,:] = sunglasses_image_png[i,j,:-1]

    # cv2.imshow("sunglasses_png", sunglasses_image_png) # check sunglasses image
    # cv2.imshow("face image", cropped_image) # check cropped image
    cv2.imshow("Output", image); # check output image

    k = cv2.waitKey(0) & 0xFF
    if k == 27: # press esc to escape
        break
    

# print(sunglasses_image_png)
# print(sunglasses_image_png.shape)
# print(rects)
# print(rect)
# print(shape.shape)
# print(x)
# print(y)
cv2.destroyAllWindows()

"""#### Detect face and put sunglasses on livestreaming"""

def preprocess_sunglasses_image(image, output_width):
    height, width, channel = image.shape # get height, width
    ratio = output_width / width
    image = imutils.resize(image, height=int(height*ratio), width=output_width)
    # image = imutils.rotate_bound(image,90)
    return image

cap = cv2.VideoCapture(0)
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects): # rects: Rectangle instance(collection of rectangles)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect) # shape: coordinates of facial landmarks
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        count = 0
        for (x, y) in shape:
            if count in {36}:
                left_eye_coordinate = (x, y)
            # draw facial landmarks
            # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            count += 1

    # draw rectangle to check face detection
    # cv2.rectangle(image, (rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0), 1)

    # make cropped image
    cropped_image = image[rect.top():rect.bottom(), rect.left():rect.right()]

    # read sunglasses.png
    # alpha channel is used to determine the transparancy of image
    sunglasses_image_png = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED) # get 3 dim data rgba (r,g,b,a:transparancy)

    # prerpocess sunglasses_image
    sunglasses_image_png = preprocess_sunglasses_image(sunglasses_image_png, rect.right()-rect.left())

    # read sunglass dimensions
    height, width, channel = sunglasses_image_png.shape
    y = left_eye_coordinate[1]-rect.top()
    # print(cropped_image.shape)
    # print(sunglasses_image_png.shape)
    # print(y)
    for i in range(height):
        for j in range(width):
            if sunglasses_image_png[i, j, 3] == 0:
                pass
            else:
                cropped_image[y-29//2+3+i,j,:] = sunglasses_image_png[i,j,:-1]

    # cv2.imshow("sunglasses_png", sunglasses_image_png) # check sunglasses image
    # cv2.imshow("face image", cropped_image) # check cropped image
    cv2.imshow("Output", image) # check output image

    k = cv2.waitKey(5) & 0xFF
    if k == 27: # press esc to escape
        break
    

# print(sunglasses_image_png)
# print(sunglasses_image_png.shape)
# print(rects)
# print(rect)
# print(shape.shape)
# print(x)
# print(y)
cv2.destroyAllWindows()
cap.release()