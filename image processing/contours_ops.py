import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

"""INFO
all information about contours you need to know
"""


def contourFeatures():
    """
    0.findContours dealing with binary imgs, be careful, the noise has a great effect on findContours, so make sure it
    is clean and without noise on the edge.

    1.moments  :return an array M which M['m00'] represents the area of the img

    2.countArea  :return area

    3.countArcLength  :return premier

    4.contour approximation  :return cnt

    5.convex hull  :return cnt @parameter:returnPoints by default is true

    6.checking is convexity  :return bool

    7.bounding rectangle{ a. straight bounding rectangle b. rotated bounding rectangle}
    the former returns x,y,w,h (using cv.rectangle),
    and the latter returns a struct like (center (x,y), (width, height), angle of rotation) \
    then using cv.boxPoints(the struct mentioned above) to get a cnt

    8.minimum enclosing circle  :return (x,y),radius

    9.fitting an ellipse  :return a struct which can be directly used in cv.ellipse(img,ellipse,color,fontSize)

    10.fitting a line  :return [vx,vy,x0,y0] where vx and vy are normalized vector (we can get angle from this)
    while (x0,y0) is a point truly on the line. Then we use some mathematics trick and cv.line to draw the whole line

    :return: None
    """

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    img = cv.imread("../Resources/triangle.png")
    imgOri = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, imgBinary = cv.threshold(img, 240, 255, cv.THRESH_BINARY_INV)
    img = cv.erode(imgBinary, kernel)
    img = cv.medianBlur(img, 9)
    # img is binary
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    """suppose we have know there is only one contour and in fact it is"""

    """moments 1.count area 2.count length"""
    # contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     cv.drawContours(imgOri, [cnt], 0, (0, 255, 0), 3)
    # cv.imshow('temp', imgOri)
    # cv.waitKey(0)
    # cnt = contours[0]
    # M = cv.moments(cnt)
    # print(M)
    # print("cx is {}, cy is {}".format(M['m10'] / M['m00'], M['m01'] / M['m00']))
    # print("area is {}".format(M['m00']))

    """count length && approximation"""
    # print(cv.arcLength(cnt, True))
    # imgApproximation = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
    cv.drawContours(imgOri, contours, 0, [255, 255, 0], 2)
    cv.imshow("tep", imgOri)
    cv.waitKey(0)

    """imgHull"""
    # imgHull = cv.convexHull(cnt)
    # cv.drawContours(imgOri, [imgHull], 0, (0, 255, 255), 3)
    # cv.imshow("temp", imgOri)
    # cv.waitKey(0)

    """bounding rect/rotated rect"""
    # x, y, w, h = cv.boundingRect(cnt)
    # cv.rectangle(imgOri, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # rotatedRectPoints = cv.minAreaRect(cnt)
    # rotatedRectPointsCnt = np.int0(cv.boxPoints(rotatedRectPoints))
    # """attention convert (np.int0) is necessary """
    # cv.drawContours(imgOri, [rotatedRectPointsCnt], 0, (0, 255, 0), 2)
    # cv.imshow('temp', imgOri)
    # cv.waitKey(0)

    """minAreaCircle"""
    # (x, y), radius = cv.minEnclosingCircle(cnt)
    # x, y, radius = round(x), round(y), round(radius)
    # cv.circle(imgOri, (x, y), radius, (255, 0, 0), 2)

    """minAreaEllipse"""
    # minAreaEllipseStruct = cv.fitEllipse(cnt)
    # cv.ellipse(imgOri, minAreaEllipseStruct, color=(0, 0, 0), thickness=2)
    # cv.imshow("temp", imgOri)
    # cv.waitKey(0)
    """fitting a line"""
    """
    @param1: cnt
    @param2: distType such as l2,l12 norm
    @param3: Numerical parameter ( C ) for some types of distances.
             If it is 0, an optimal value is chosen.
    @param4 reps: Sufficient accuracy for the radius (distance between the coordinate origin and the line).
    @param5 aeps: Sufficient accuracy for the angle.
    0.01 would be a good default value for reps and aeps.
    """
    # (vx, vy, x0, y0) = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
    # rows, cols = img.shape
    # lefty = (-x0 * vy / vx) + y0
    # righty = vy / vx * (cols - 1 - x0) + y0
    # cv.line(imgOri, (0, lefty), (cols - 1, righty), (0, 255, 0), 2)
    # cv.imshow('temp', imgOri)
    # cv.waitKey(0)


def contourProperties():
    """
    @properties:
    1.aspect ration  :width/height
    2.extent  :true area/bounding rect area
    3.solidity  :true area/hull area
    4.equivalent diameter is the diameter of the circle whose area is same as the contour area.
    equivalent diameter = sqrt(4*area/pi)
    5.orientation  :(x,y),(Major Axis,minor axis),angel=cv.fitEllipse(cnt)
    6.min_val,max_val,min_loc,max_loc=cv.minMaxLoc(img,mask=mask)
    7. Mean color or Mean intensity
    the average color of an object. Or it can be average intensity of the object in grayscale
    8.extreme points Extreme Points means topmost, bottommost, rightmost and leftmost points of the object
    topmost= cnt[cnt[:,:,0].argmin()][0] for example cnt.shape=(nsamples,1,2], where 2 stands for x axis(0),
    and 1 stands for y axis
    leftmost = cnt[cnt[:,:,0].argmin()][0]
    rightmost = cnt[cnt[:,:,0].argmax()][0]
    topmost = cnt[cnt[:,:,1].argmin()][0]
    bottommost = cnt[cnt[:,:,1].argmax()][0]
    """


def contourFunctions():
    """
    1.cv.convexityDefects(cnt,hull) attention: hull must be indices which can be got from
    cv.convexHull(cnt,returnPoints=False),and it returns a list. A value of this list is like
    [start point,end point,farthest point,approximation dist], the first three values are also
    indices

    2.cv.pointPolygonTest(cnt,(pointTestx,pointTesty),Ture),the third value determines whether it
    computes distance of the farthest point to its nearest cnt. if False it will just return a sign
    like -1;0;1 point inside cnt ,outside, and on respectively. if you don't want to know the dist,
    just let the third param to be False to save time.

    3.match shapes: dealing grayscale imgs cv.matchShapes(cnt1,cnt2,method,parameter)
    parameter can be set to be 0.0, and method can be cv.CONTOURS_MATCH_l1
                                                      cv.CONTOURS_MATCH_l2
                                                      and similarly .._l3
    it returns the difference degree, more similar the smaller
    @# 3 using Hu moments to compute similarity and so far i don't know how and why to compute
    :return:
    """
    img = cv.imread('../Resources/cross img.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(imgGray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    hull = cv.convexHull(cnt, returnPoints=False)
    defects = cv.convexityDefects(cnt, hull)
    cv.drawContours(img, contours, -1, (0, 255, 0), 1)
    for perDefect in defects:
        [b, e, f, d] = perDefect[0]
        cv.line(img, tuple(cnt[b][0]), tuple(cnt[e][0]), (0, 255, 255), 2)
        cv.circle(img, tuple(cnt[f][0]), 3, (0, 0, 255), 1)
    cv.imshow("temp", img)
    cv.waitKey(0)
    # print(cv.pointPolygonTest(cnt, (20, 20), False))
    print(cv.matchShapes(cnt, cnt, method=cv.CONTOURS_MATCH_I1, parameter=0))


def contoursHierarchy():
    """
    sometime we want to describe a relation between one contour and another just like a relation between
    a farther and his son. So we have hierarchy. Different retrieve mode select different contours.
    for example:
        cv.RETRE_LIST: it ignore the relation in contours
        It simply retrieves all the contours, but doesn't create any parent-child relationship.
        cv.RETRE_CCOMP: it only has two level(relative relation)
        cv.RETRE_EXTREME: it selects only the extreme outer contours
        cv.RETRE_TREE: This flag retrieves all the contours and arranges them to a 2-level hierarchy
    :return:
    """


if __name__ == "__main__":
    contourFeatures()
    # contourFunctions()
    pass
