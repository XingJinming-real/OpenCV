import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def Shi_Tomasi_corner_detection():
    """
    this function is more appropriate for tracking

    it finds the most prominent corners in an img
    it has non-maximum suppression in 3*3 local area
    it is similar to harris while it changes R into R=min(r1,r2) from R=det(M)-k(trace(M))
    it deals with grayScale img
    @:parameter (img,cornersNum,cornersScore(0~1),min L2 between arbitrary two corners)
    :return:
    """
    img = cv.imread('../Resources/answerCard.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(np.float32(imgGray), 10, 0.8, 20)
    for perCorner in corners:
        cv.circle(img, tuple(perCorner.ravel()), 3, [0, 0, 255], -1)
    cv.imshow('temp', img)
    cv.waitKey(0)


Shi_Tomasi_corner_detection()
