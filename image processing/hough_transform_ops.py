import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""more about HoughLines and it's realLife application please view detection_of_road_edges.py"""


def empty(x):
    pass


cv.namedWindow('result', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('param1', 'result', 75, 255, empty)
cv.createTrackbar('param2', 'result', 40, 255, empty)


def houghTransformCircles():
    """attention:
        img should be grayScale/binary
        :return circles circles.shape=[1,n,3] where n stands for how many circles it has detected.
        so, you should circle=circle[0,:], then circle would be a list containing [X0,Y0,radius]
        :parameters
                    method: cv.HOUGH_GRADIENT and cv.HOUGH_GRADIENT_ALT
                    dp: often(1) Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1,
                        the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
                        For HOUGH_GRADIENT_ALT the recommended value is dp=1.5, unless some small very circles need to be detected.
                    minDist: the minLength among centers of all circles
                    param1:  the upperThreshold in Canny, if you use cv.HOUGH_GRADIENT_ALT, you may be set param1 to be higher
                             like 300
                    param2: the accumulator threshold for a polygon to be considered as a circle, for method cv.HOUGH_GRADIENT
                            it returns the pixels num, and for another ...ALT, it returns the perfectness,
                            The closer it to 1, the better shaped circles algorithm selects. In most cases 0.9 should be fine.
                            If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
                            But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.
                    minRadius: minimum circle radius
                    maxRadius: maximum circle radius if <=0 uses the maximum img diameter.
                               if<0 HOUGH_GRADIENT returns centers without radius, ...ALT always returns radius.
    """
    img = cv.imread('../Resources/openCvlogo.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    while True:
        imgResult = img.copy()
        param1 = cv.getTrackbarPos('param1', 'result')
        param2 = cv.getTrackbarPos('param2', 'result')
        circles = cv.HoughCircles(imgGray, cv.HOUGH_GRADIENT, 1, 10,
                                  param1=param1, param2=param2, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(imgResult, (i[0], i[1]), i[2], (255, 255, 255), 2)
            # draw the center of the circle
            cv.circle(imgResult, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv.imshow('result', imgResult)
        cv.waitKey(22)


houghTransformCircles()
