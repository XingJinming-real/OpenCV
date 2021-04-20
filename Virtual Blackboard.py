import cv2
import numpy as np

Cap = None
img = None
exitedPoints = []


def empty(a):
    print(a)
    pass


cv2.namedWindow('TrackBars', cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('hime', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('hMin', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('hMax', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('sMin', 'TrackBars', 156, 255, empty)
cv2.createTrackbar('sMax', 'TrackBars', 255, 255, empty)
cv2.createTrackbar('vMin', 'TrackBars', 88, 255, empty)
cv2.createTrackbar('vMax', 'TrackBars', 255, 255, empty)


def getBeginPoint(mask):
    x = 0
    y = 0
    w = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        epsilon = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.001 * epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
    return x, y, w


def drawCircle():
    for perPoint in exitedPoints:
        cv2.circle(img, (perPoint[0] + perPoint[2] // 2, perPoint[1]), 3, (0, 0, 0), 10, cv2.FILLED)


def main():
    global Cap, img, exitedPoints
    Cap = cv2.VideoCapture(0)
    success, img = Cap.read()
    while True:
        success, img = Cap.read()
        if not success or cv2.waitKey(1) & 0xFF == ord('q'):
            break
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hMin = cv2.getTrackbarPos('hMin', "TrackBars")
        hMax = cv2.getTrackbarPos('hMax', "TrackBars")
        sMin = cv2.getTrackbarPos('sMin', "TrackBars")
        sMax = cv2.getTrackbarPos('sMax', "TrackBars")
        vMin = cv2.getTrackbarPos('vMin', "TrackBars")
        vMax = cv2.getTrackbarPos('vMax', "TrackBars")
        mask = cv2.inRange(imgHSV, lowerb=np.array([hMin, sMin, vMin]), upperb=np.array([hMax, sMax, vMax]))
        cv2.waitKey(0)
        x, y, w = getBeginPoint(mask)
        exitedPoints.append([x, y, w])
        drawCircle()
        cv2.imshow('imgResult', img)
        cv2.waitKey(2)


if __name__ == '__main__':
    main()
