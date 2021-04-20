import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""more about HoughLines and it's realLife application please view Detection of Road Edges.py"""


def empty(x):
    print(x)


cv.namedWindow('result', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('BinaryThreshold', 'result', 240, 255, empty)


def houghTransform():
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
    # img = cv.imread('Resources/triangle.png')
    img = cv.imread('Resources/openCvlogo.png')
    # imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    while True:
        imgResult = img.copy()
        # imgCanny = cv.Canny(imgGray, 80, 180)
        __, imgBinary = cv.threshold(imgGray, 20, 255, cv.THRESH_BINARY)
        # lines = cv.HoughLinesP(imgCanny, 1, np.pi / 180, 100, maxLineGap=40)
        # if lines is None:
        #     break
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv.line(imgResult, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # cv.imshow('result', imgResult)
        # cv.waitKey(22)
        cv.imshow('img', imgBinary)
        cv.waitKey(0)
        circles = cv.HoughCircles(imgBinary, cv.HOUGH_GRADIENT, 1, 10,
                                  param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv.imshow('detected circles', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # for circle in circles:
        #     pass


houghTransform()
