import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

"""
    filtering is mainly fulfilled by low-pass filter
    opencv provides cv.filter2D(img,depth,kernel)
    kernel (given by users)
    1.avgFilter blur/box filter
    2.Gaussian blur  perfectible for gaussian noise
    3.median blur  perfectible for salt-and-pepper noise
    4.bilateral filter  (slower than other filters)
    
    # bilateralFilter is highly effective in noise removal while keeping edges sharp. Details are as follows
    # We already saw that a Gaussian filter takes the neighbourhood around the pixel and finds its Gaussian weighted average. 
    # This Gaussian filter is a function of space alone, that is, nearby pixels are considered while filtering. 
    # It doesn't consider whether pixels have almost the same intensity. 
    # It doesn't consider whether a pixel is an edge pixel or not. 
    # So it blurs the edges also, which we don't want to do.
    # Bilateral filtering also takes a Gaussian filter in space, but one more Gaussian filter which is a function of pixel difference. 
    # The Gaussian function of space makes sure that only nearby pixels are considered for blurring, 
    # while the Gaussian function of intensity difference makes sure that only those pixels 
    # with similar intensities to the central pixel are considered for blurring. So it preserves the edges 
    # since pixels at edges will have large intensity variation.

    """


def main():
    img = cv.imread('Resources/openCvlogo.png')
    kernel2D = np.ones((3, 3)) / (3 * 3)
    img2D = cv.filter2D(img, -1, kernel2D)
    imgAvg = cv.boxFilter(img, -1, (3, 3))
    imgGaussian = cv.GaussianBlur(img, (7,7), 0)
    # otherwise get a gaussianKernel by cv.getGaussianKernel()
    imgMedian = cv.medianBlur(img, 5)
    """
    cv.bilateralFilter(src,d(diameter of neighbours to be computed),sigmaColor,sigmaSpace
    sigmaSpace=the space of pixels to be computed (the larger more pixels will be included)
    sigmaColor=determine the upperBound of the 'semi-similar color' pixels
    if diameter>0,sigmaSpace will be diameter
    if diameter<0 diameter is the same with sigmaSpace
    """
    # For simplicity, you can set the 2 sigma values to be the same.
    # If they are small (< 10), the filter will not have much effect, whereas if they are large (> 150),
    # they will have a very strong effect, making the image look "cartoonish".
    # Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications,
    # and perhaps d=9 for offline applications that need heavy noise filtering.
    imgBilateral = cv.bilateralFilter(img, 9, 75, 75)
    figure, ax = plt.subplots(1, 5, figsize=(17, 17))
    nameList = ["2D(using avgKernel)", 'Avg', 'Gaussian', 'median', 'bilateral']
    imgList = [img2D, imgAvg, imgGaussian, imgMedian, imgBilateral]
    for i in range(5):
        ax[i].imshow(imgList[i])
        ax[i].set_title(nameList[i])
    plt.show()


if __name__ == '__main__':
    main()
