import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

"""
    we know that in most cases pixel p is a corner if on a neighbourhood circle of this pixel
there are more than n pixels that brighter or darker(>p+t,or p-t, here p is a threshold given
by users) than the center pixel. here n is chosen to be 12.
However it's slow, so we apply a high speed test before we use this fast algorithm. Details are
as follows. 
    first, if p is a corner, at least three of four pixels(straight right, straight left, straight up
and straight down) are darker or brighter than the center pixel p. through this, we reject as many
false candidates, then we apply the full segments of fast algorithm.
however, 1. it doesn't reject candidates that n<12 
         2. it's not optimal depending on the order of the circles' pixels
         3. the result of high speed test is thrown away
         4. many corners are adjacent
          
for 1,2,3 we use ID3(decision tree using information entropy) to handle.
    4 we use non-maximal suppression"""


def main():
    img = cv.imread('../Resources/corner.png')
    # first we create a fast object
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    __,imgBinary=cv.threshold(imgGray,220,255,cv.THRESH_BINARY_INV)
    fast = cv.FastFeatureDetector_create()
    # lets see the default parameters
    fast.setNonmaxSuppression(0)
    fast.setThreshold(20)
    kp = fast.detect(imgBinary, None)
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    img = cv.drawKeypoints(img, kp, None, color=(0, 0, 0))
    cv.imshow("result", img)
    cv.waitKey(0)


main()
