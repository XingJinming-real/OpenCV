import numpy as np
import cv2

"""half processed"""


def getBiggestPoints(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggestArea = 0
    points = []
    for cnt in contours:
        epsilon = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02 * epsilon, True)
        if len(approx) == 4:
            if area > biggestArea:
                biggestArea = area
                points = approx
    return points


def getGoalPst(biggestPoints):
    biggestPointsNew = np.zeros_like(biggestPoints)
    biggestPointsNew[3] = biggestPoints[2]
    biggestPointsNew[1] = biggestPoints[0]
    biggestPointsNew[2] = biggestPoints[3]
    biggestPointsNew[0] = biggestPoints[1]
    return biggestPointsNew


def getPerspectiveImg(biggestPoints, img):
    biggestPoints = np.float32(np.resize(biggestPoints, (4, 2)))
    print(biggestPoints)
    width, height = img.shape[:2]
    # goalPst = getGoalPst(biggestPoints)
    # print(goalPst)
    # input()
    M = cv2.getPerspectiveTransform(biggestPoints, np.float32([[0, 0], [300, 0], [0, 300], [300, 300]]))
    imgPerspective = cv2.warpPerspective(img, M, (width, height))
    return imgPerspective


def getImg():
    img = cv2.imread('../Resources/answerCard.png')
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.GaussianBlur(imgGray, (3, 3), 1)
    # imgGray = cv2.dilate(imgGray, kernel=np.ones((3, 3)), iterations=1)
    # imgGray = cv2.erode(imgGray, kernel=np.ones((3, 3)), iterations=1)
    imgCanny = cv2.Canny(imgGray, 70, 200)
    cv2.imshow("imgCanny", imgCanny)
    cv2.waitKey(0)
    biggestPoints = getBiggestPoints(imgCanny)
    x, y, w, h = cv2.boundingRect(biggestPoints)
    imgWithRect = img.copy()
    cv2.rectangle(imgWithRect, (x, y), (x + w, y + h), (0, 0, 255), 3)
    imgPerspective = getPerspectiveImg(biggestPoints, img)
    return img, imgGray, imgCanny, biggestPoints, imgWithRect, imgPerspective



if __name__ == '__main__':
    main()
