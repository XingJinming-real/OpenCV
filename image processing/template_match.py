import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def empty(x):
    print(x)


cv.namedWindow('TrackBar', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('threshold(*100)', 'TrackBar', 80, 100, empty)


def templateMatch():
    """
    cv.templateMatch(img,template,method)
    attention: if use cv.TM_SQDIFF the best matched patch will be the lowest value in the return img(the darkest point)

    template can be a color img which is same to the img
    :return: a grayscale image,
    where each pixel denotes how much does the neighbourhood of that pixel match with template.
    """
    img = cv.imread('../Resources/Mario.png')
    imgGray = img.copy()
    imgTemplate = cv.imread('../Resources/Mario_coin.png')
    rows, cols, channels = imgTemplate.shape
    res = cv.matchTemplate(imgGray, imgTemplate, cv.TM_CCOEFF_NORMED)
    while True:
        imgTemp = img.copy()
        threshold = cv.getTrackbarPos("threshold(*100)", 'TrackBar') / 100
        resThreshold = np.where(res >= threshold)  # return (row,col) so in the cv.rectangle you should reverse it
        for perPoint in zip(*resThreshold[::-1]):
            cv.rectangle(imgTemp, perPoint, (perPoint[0] + cols, perPoint[1] + rows), (0, 0, 255), 1)
        cv.imshow("TrackBar", imgTemp)
        cv.waitKey(22)
    pass


templateMatch()
