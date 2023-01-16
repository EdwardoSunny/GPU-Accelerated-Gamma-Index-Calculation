import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2, math
import numpy as np
import time
from numba import njit
from numba import cuda
from numba import vectorize
from numba.typed import List
from interpolation import interp
# assumes images are same resoltion/size

# this only works for square, full images (pixels values are not transparent, etc.)
# image geometry
# pixel wise:
# 0 1 2 3 4 5 x
# 0 1 2 3 4 5
# 0 1 2 3 4 5
# ...
# y
# [y][x]

# real units (mm) wise
# 0 1 2 3 4 5 x
# 1 2 3 4 5 6
# 2 3 4 5 6 7
# 3 4 5 6 7 8
# y
# [y][x]

# definition of 0mm is at the left edge of the first pixel

# read as gray scale
reference = cv2.imread('ref.png', 0) # reference image
test = cv2.imread('test.png', 0) # test image
print(type(reference))
#
spacing = 0.1 # pixel spacing, scaling between pixel and real life, real units/pixel (e.g. mm/pixel)
search_radius = 1 # real units (e.g. mm)
search_percent = 0.1 # decimal standing for percent
radial_step_size = 1 # real units (e.g. mm)
angular_step_size = 1 # degrees

gammaImage = np.zeros((reference.shape[0], reference.shape[1]))
print(gammaImage.shape)
# gpu_gamma_image = cuda.jit

#@njit
# @cuda.jit(device=True)
def get_interp_image_x_y(xRange, yRange):
    xData = np.zeros(xRange)
    yData = np.zeros(yRange)
    for i in range (1, xRange+1):
        xData[i-1] = i
    for i in range(1, yRange+1):
        yData[i-1] = i
    return [xData, yData]

# ref pos is a list [x, y] from original image
# computes gamma by interating test image
#@njit

# @vectorize(['float32(float32)'], target='cuda')
def get_2D_gamma_full_for_one_pixel(refPos):
    gammaList = List()
    # range of x values
    testXYData = get_interp_image_x_y(len(test[0]), len(test))
    # interp data
    testXData = testXYData[0] * spacing
    testYData = testXYData[1] * spacing
    testZData = test

    # interpFunction = interp(testXData, testYData, testZData, kind='linear')

    refVal = float(reference[refPos[1]][refPos[0]])
    xRefRealPos = float(refPos[0]*spacing)
    yRefRealPos = float(refPos[1]*spacing)

    # print("ref" + str(xRefRealPos))
    # print(yRefRealPos)


    # in real units (mm)

    rCount = 0
    thetaCount = 0
    while (rCount < search_radius+radial_step_size):
        print(rCount)
        while (thetaCount < 360+angular_step_size):
            # starting with the positive x axis, ccw
            xTestBasedOnStartPos = rCount * math.cos(math.radians(thetaCount)) * spacing
            # multiply -1 to redfine stupid coord sys
            yTestBasedOnStartPos = rCount * math.sin(math.radians(thetaCount)) * -1 * spacing


            # refPos is where the current reference pixel is, since
            # the calculations above is relative to where the reference pixel is,
            # must add to find where it actually is (localize the vector)

            xTestRealPos = xTestBasedOnStartPos + xRefRealPos
            yTestRealPos = yTestBasedOnStartPos + yRefRealPos
            if not(xTestRealPos < 0 or xTestRealPos > len(testXData)-1 or yTestRealPos < 0 or yTestRealPos > len(testYData)-1):
                # if (rCount == radial_step_size):
                #     print(thetaCount)
                currVal = interp(testXData, testYData, testZData, xTestRealPos, yTestRealPos)
                # maybe this is wrong
                currentToTestDistance = abs(math.sqrt(((xRefRealPos - xTestRealPos) ** 2) + ((yRefRealPos - yTestRealPos) ** 2)))
                currGamma = math.sqrt((((currentToTestDistance) ** 2) / search_radius) + ((refVal - currVal) ** 2) / search_percent)
                gammaList.append(currGamma)
            thetaCount += angular_step_size
            rCount += radial_step_size        
    
    return gammaList

def get_zero_matrix(n, m):
    zeros = []

    for i in range(0, m):
        tempVector = []
        for m in range(0, n):
            tempVector.append(0)
        zeros.append(tempVector)

def get_gamma_image():
    for y in range(0, len(reference)):
        currentRow = []
        for x in range(0, len(reference[0])):
            print("x: " + str(x) + " y: " + str(y))
            currFullGamma = get_2D_gamma_full_for_one_pixel([x, y])
            currGamma = min(currFullGamma)
            currentRow.append(currGamma)
        gammaImage[y] = currentRow
    return gammaImage

def get_passing_rate():
    totalPass = 0
    for gammaRow in gammaImage:
        for gammaVal in gammaRow:
            if (gammaVal <= 1):
                totalPass += 1
    passDecimal = totalPass/(len(gammaImage) * len(gammaImage[0]))
    return str(passDecimal * 100) + '%'

def main():
    #gammaImage = get_gamma_image()
    st = time.time()
    gammaImage = get_gamma_image()
    et = time.time()
    print("calc time: " + str(et-st))
    plt.figure("Original Reference Image")
    npReferenceImageArr = np.array(reference)
    imgOriginalReference = plt.imshow(npReferenceImageArr, cmap='gray')

    plt.figure("Original Test Image")
    npTestImageArr = np.array(test)
    imgTest = plt.imshow(npTestImageArr, cmap='gray')

    plt.figure("Gamma Image")
    npGammaImageArr = np.array(gammaImage)
    imgGamma = plt.imshow(npGammaImageArr, cmap='gray')
    passingRate = get_passing_rate()
    plt.text(-2, -1, 'Passing Rate: ' + str(passingRate), bbox=dict(fill=False, edgecolor='red', linewidth=3))


    plt.show()



if (__name__ == '__main__'):
    main()
