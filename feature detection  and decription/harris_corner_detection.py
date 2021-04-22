import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def harris_corner_detection():
    """corners are regions in the image with large variation in intensity in all the directions"""
    """deal with grayScale img"""
    """with some mathematical tricks we can get R=det(M)-k(trace(M))^2, where M is sum(w(x,y)[IxIx,IxIy
                                                                                            IxIy,IyIy])
                                                                                            w(x,y) is weighted matrix"""
    """det(M)=lambda1(r1)lambda2(r2) and trace(M)=r1+r2
    r1 and r2 are the eigenvalues of M
    when |R| is small which means r1 and r2 is small the region is flat
    when R<0 which happens when r1>>r2 or vice versa, the region is edge
    when R is large, which means r1 and r22 are large and r1≈r2 the region is corner"""
    """:parameter img(float 32),blockSize(the neighbour considered for corner detection
    ksize: aperture parameter of the sobel derivative used.
    k: harris detector free parameter"""
    """it returns the score resultImg with the same of ori """
    img = cv.imread('../Resources/answerCard.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGray = np.float32(imgGray)
    imgHarris = cv.cornerHarris(imgGray, 2, 3, 0.04)
    """the value in imgHarris is the R mentioned above"""
    tempI = cv.cornerEigenValsAndVecs(imgGray, 2, 3)
    """tempI return(x,y,6) where 6 are
        λ1,λ2 are the non-sorted eigenvalues of M
        x1,y1 are the eigenvectors corresponding to λ1
        x2,y2 are the eigenvectors corresponding to λ2"""
    (x, y) = np.where(imgHarris > 0.2 * imgHarris.max())
    for perPixel in list(zip(y, x)):
        cv.circle(img, perPixel, 2, [0, 255, 0], -1)
    """if we want to get more precise result we can use cv.cornerSubPix(img(gray,float),centroids,
    half-radius of windowSize, half-radius of zero zone ( for singular situation of windowSize)
    that under this zero zone, pixels will not be summed"""
    """first we should get centroids through cv.connectedComponentsWithStats(we also use cv.connectedComponents in fg
    extraction named watershed algorithm), which returns
    ret,labels,stats(statistics output for each label, including the background label. Statistics are accessed via
    stats(label, COLUMN) where COLUMN is one of ConnectedComponentsTypes)
    ,centroids.
    It iterates for better results until reaches a threshold we set.
    """
    __, imgHarris = cv.threshold(imgHarris, 0.01 * imgHarris.max(), 255, cv.THRESH_BINARY)
    # !!!we need binary img to filter noise
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100,
                1)  # the first parameter means the algorithm returns
    # when any one of them(max-iter,precession satisfies)
    # count（min iterNum) ,max_iter(Num),eps(returns when reach 0.001 precession)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(np.uint8(imgHarris))
    corners = cv.cornerSubPix(imgGray, np.float32(centroids), (1, 1), (-1, -1), criteria)
    for perPixel in corners:
        cv.circle(img, tuple(perPixel), 1, [0, 0, 255], -1)
    cv.imshow('temp', img)
    cv.waitKey(0)
    """ we can see the red points(subPixels) is better than the green points"""


harris_corner_detection()
