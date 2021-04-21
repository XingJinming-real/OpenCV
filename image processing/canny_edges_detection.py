import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def empty(x):
    pass


cv.namedWindow('temp', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('minValue', 'temp', 0, 255, empty)
cv.createTrackbar('maxValue', 'temp', 0, 255, empty)

cv.namedWindow('temp1', cv.WINDOW_AUTOSIZE)


def main():
    img = cv.imread('../Resources/answerCard.png', 0)
    while True:
        lowerBound = cv.getTrackbarPos('minValue', 'temp')
        upperBound = cv.getTrackbarPos('maxValue', 'temp')
        imgCanny = cv.Canny(img, lowerBound, upperBound, L2gradient=True)
        kernelMor = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        imgDilation = cv.dilate(imgCanny, kernelMor, iterations=2)
        cv.imshow('temp1', imgDilation)
        cv.waitKey(1)
        cv.imshow('temp', imgCanny)
        cv.waitKey(1)


main()
