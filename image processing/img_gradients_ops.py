import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    """
    the so-called gradient is computed by high-pass filters.Here i will introduce 3 filters
    1.sobel
            [-1,0,1
            -3,0,3
            -1,0,1]
    2.Scharr
            [-3,0,3
            -10,0,10
            -3,0,3]
    3.Laplacian
            [0,1,0
            1,-4,1
            0,1,0]
    :return:
    """

    def getGradient():
        # be attention if the ddepth is set to be cv.CV_8U,then you will lose one side edge.
        # because when compute the gradient from black to white you will get the right answer.
        # when you compute the gradient from white to black the value is negative but the type is
        # uint8, unable to denote negative value,so the outcome will be 0
        # the default kernel size in sobel is scharr as shown above

        """be attention if the original img is uint8, and you convert it to cv_CV32_F, then the img will to wrong
            you need to convert it back to uint8"""
        img = cv.imread('../Resources/openCvlogo.png', 0)
        plt.imshow(img, 'gray', vmin=0, vmax=255)
        imgSobelX = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=5)
        imgSobelY = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=5)
        imgSobel = cv.Sobel(img, cv.CV_32F, 1, 1, ksize=5)
        imgScharrX = cv.Sobel(img, cv.CV_32F, 1, 0)
        imgScharrY = cv.Sobel(img, cv.CV_32F, 0, 1)
        imgScharr = cv.Sobel(img, cv.CV_32F, 1, 1)
        imgLaplace = cv.Laplacian(img, cv.CV_32F)
        figure, ax = plt.subplots(2, 4, figsize=(14, 7))
        imgList = [img, imgSobelX, imgSobelY, imgSobel, imgScharrX, imgScharrY, imgScharr, imgLaplace]
        for i in range(len(imgList)):
            imgList[i] = np.uint8(imgList[i])
        nameList = ['ori', 'sobelX', 'sobelY', 'sobel', 'scharrX', 'scharrY', 'scharr', 'Laplace']
        for i in range(8):
            ax[round(i / 8)][int(i % 4)].imshow(imgList[i], 'gray', vmin=0, vmax=255)
            ax[round(i / 8)][int(i % 4)].set_title(nameList[i])
        ax[1][0].imshow(img, 'gray', vmin=0, vmax=255)
        ax[1][0].set_title('Ori')
        plt.show()

    getGradient()


if __name__ == "__main__":
    main()
