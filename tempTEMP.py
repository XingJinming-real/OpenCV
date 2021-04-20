import numpy as np
import cv2 as cv
import time

def empty(x):
    pass


# cap = cv.VideoCapture("Resources/03.avi")
cv.namedWindow('result', cv.WINDOW_AUTOSIZE)
# cv.createTrackbar('CannyLowerB', 'img', 0, 255, empty)
# cv.createTrackbar('CannyUpperB', 'img', 0, 255, empty)
# cv.createTrackbar('CannyUpperC', 'img', 0, 255, empty)
# cv.createTrackbar('ThresholdAfterCanny', 'img', 0, 255, empty)
img = cv.imread("Resources/HoughLine.png")
# cv.createTrackbar('binaryThreshold', 'result', 222, 255, empty) for 01.avi and 02.avi
cv.createTrackbar('binaryThreshold', 'result', 200, 255, empty)  # for 03.avi
cv.createTrackbar('houghThreshold', 'result', 20, 255, empty)
cv.createTrackbar('minLength', 'result', 265, 400, empty)
cv.createTrackbar('maxLenGap', 'result', 40, 400, empty)
cv.createTrackbar('lowerBound', 'result', 60, 255, empty)
cv.createTrackbar('upperBound', 'result', 150, 255, empty)

imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('tepm',imgHSV)
cv.waitKey(0)
while True:
    # if not cap.isOpened():
    #     break
    # ret, img = cap.read()
    # if not ret:
    #     break
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgResult = img.copy()
    lowerBound = cv.getTrackbarPos('lowerBound', 'result')
    upperBound = cv.getTrackbarPos('upperBound', 'result')
    binaryThreshold = cv.getTrackbarPos('binaryThreshold', 'result')
    houghThreshold = cv.getTrackbarPos('houghThreshold', 'result')
    minLength = cv.getTrackbarPos('minLength', 'result')
    maxLenGap = cv.getTrackbarPos('maxLenGap', 'result')
    # cv.imshow("imgGray", imgGray)
    # imgThreshold = cv.adaptiveThreshold(imgCanny, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # cv.imshow('img', imgCanny)
    rows, cols = imgGray.shape
    ROI = imgGray.copy()
    ROI[:60, :] = np.zeros((60, cols))
    __, ROI2 = cv.threshold(ROI, binaryThreshold, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    ROI3 = cv.morphologyEx(ROI2, cv.MORPH_CLOSE, kernel)
    ROI3 = cv.morphologyEx(ROI3, cv.MORPH_DILATE, kernel)
    ROI4 = cv.Canny(ROI3, lowerBound, upperBound)

    Lines = cv.HoughLinesP(ROI4, 1, np.pi / 180, houghThreshold, minLineLength=minLength, maxLineGap=maxLenGap)
    if Lines is None:
        continue
    for line in Lines:
        x1, y1, x2, y2 = line[0]
        # if np.abs(x1 - x2) <= 10 or np.abs(y1 - y2) <= 10 or (x2 - x1) / (
        #         y1 - y2) > 1 or (x2 - x1) / (y2 - y1) > 0.8 or (
        #         np.abs(x1 - x2) < 20 and np.abs(y1 - y2) > 150):
        #     continue#


        print("x1={},y1={}\nx2={},y2={}".format(x1, y1, x2, y2))
        cv.line(imgResult, (x1, y1), (x2, y2), (0, 0, 255), 3)
    print(20 * '*')
    # cv.imshow('img', imgHSV)
    cv.imshow("result", imgResult)
    # mask = cv.inRange(imgHSV, np.array([0, 220, 220]), np.array([20, 255, 255]))
    # cv.imshow("mask", mask)
    # imgTemp = cv.bitwise_and(img, img, mask=mask)
    # cv.imshow('img', imgTemp)
    cv.waitKey(20)
