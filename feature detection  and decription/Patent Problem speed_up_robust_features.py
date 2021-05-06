import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""暂时未充分了解原理
SURF is good at handling images with blurring and rotation, 
but not good at handling viewpoint change and illumination change.
"""


def surf():
    img = cv.imread('../Resources/corner.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # here we set hessian threshold to 400
    # in practice this threshold is usually set to 300-500
    # all potential interest points that values are under threshold will be rejected
    """here's a patent problem, so we skip this chapter"""
    surfDetector = cv.xfeatures2d.SURF_create(400)
    print(surfDetector.getUpright())

    kp = surfDetector.detect(imgGray)
    des = surfDetector.compute(img, kp)
    cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("temp", img)
    cv.waitKey(0)


surf()
