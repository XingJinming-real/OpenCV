import cv2 as cv
import numpy as np

"""如果要分割的物体之间相互接触，使用cv.distanceTransform,其返回一个
与原图像同大小的值是当前位置像素到最近白色像素点的距离的矩阵
如果要分割的物体没有相互接触，仅使用普通Threshold即可"""


def imgSegmentation():
    """注意我们使用watershed算法，处理灰度图
    首先我们得到一定是foreGround的图像(binary，我们想要让不同的物体分开))(该图像中，sure-foreground为255，其余为0
    然后得到一定是background的图像(binary)(该图像中，褐色部分一定是background)
    除此之外，我们还剩下我们不确定的区域(不确定的区域可以通过backgroundImg-foreground，注意
    他们通常是物体之间的交界处，)
    原始的watershed算法容易过检测，因此我们使用带有标签标记的图像(通过cv.connectedComponents(sure-foreground(binary))(该函数
    返回marker，marker是将背景(全黑)标记为0，其余为从1开始递增)。
    通过上述cv.connectedComponents()得到marker后，再用于cv.watershed之前，我们要求背景标签一定是-1，
    故我们使用marker=marker+1,然后用原先求得的marker[sure_bg-sure_fg==255]=-1进行更改
    这样我们就得到了要求的marker，使用cv.watershed(img(在原图上作画),marker),使用完该函数后，作为参数的marker将会改变，
    marker矩阵中边界将会变为-1，这时，我们的marker在原图边界处的值就为-1，此时只需要将img[marker=-1]=[0,0,255]就可以在原图像上标出检测的结果
    marker==-1处就是我们的物体的分割边界
    应用：
        1.物体计数
        2.待补充
    """
    img = cv.imread("../Resources/imgSegmentation.png")
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 此处我们使用大津展之OTSU自动选取阈值(对获得的直方图，找到一个阈值使得该阈值于直方图中peaks的方差最小)处理
    __, imgBinary = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # 以为我们所求的待分割图像物体连接在一起，使用distanceTransform(上文所述)较好(目的是使图像相互之间不在接触)，如果是物体分开，则不需要进行该处理

    imgDistanceTransform = cv.distanceTransform(imgBinary, cv.DIST_L2,
                                                # 现在我们去求一定是object的图像，对imgDistanceTransform进行2*open
                                                cv.DIST_MASK_5)  # 3 for roughly 5 for precise both of two are O(N)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    sure_fg = cv.morphologyEx(imgDistanceTransform, cv.MORPH_ERODE, kernel,
                              iterations=3)
    # sure_fg=cv.erode(imgDistanceTransform,cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv.morphologyEx(imgBinary, cv.MORPH_DILATE, kernel, iterations=2)
    imgUnknown = np.uint8(cv.subtract(sure_bg, sure_fg))  # 注意此处有saturate操作
    #  接着我们获得marker
    __, marker = cv.connectedComponents(sure_fg)  # bg is set to be 0 and others are set to be 1(auto_increment)
    marker = marker + 1
    marker[imgUnknown == 255] = 0  # watershed requires the unknown area to be 0
    marker = cv.watershed(img, marker)
    # after watershed, the marker will be modified. Pixels in the segmentation edge will be set to -1
    img[marker == -1] = [150, 5, 255]
    cv.imshow("result", img)
    cv.waitKey(0)


imgSegmentation()
