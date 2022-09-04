import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.interpolate import interp1d

# image geometry
# 0mm, 1mm, 2mm, 3mm, etc.
# pixel 0, pixel 1, pixel 2, etc.

# configurable constants
image1 = [10, 10, 10, 10]# [10, 10, 10, 10, 15, 10, 2, 3, 4, 6, 2.2, 1, 10, 4, 5, 6, 6, 7] # reference image
image2 = [10, 10, 10, 10]# [10, 10.3, 6, 10.1, 12, 7, 2, 3, 4, 6, 2.2, 1, 10, 4, 5, 6, 6, 7] # test image
spacing = 0.5 # pixel spacing, scaling between pixel and real life, real units/pixel (e.g. mm/pixel)
search_size = 4 # real units (e.g. mm)
search_percent = 0.1 # decimal standing for percent
step_size = 0.1 # real units (e.g. mm)
left_image_bound = 0 # set by person, draw, etc. in pixels, 0 indexing
right_image_bound = 10 # set by person, draw, etc. in pixels, 0 indexing

# static constants 
image1 = image1[left_image_bound:right_image_bound+1]
image2 = image2[left_image_bound:right_image_bound+1]
print("bounded image: " + str(image1))

gammaImage = []


def test_image_pos_to_interp_data(img):
    # [xs, ys]
    coordinateList = [[], image2]
    for i in range(1, len(img)+1):
        coordinateList[0].append(i)
    return coordinateList

def image_data_to_real_units(posData):
    for i in posData:
        i *= spacing
    return posData


def arrays_to_img(pixelVal, pixelPos):
    pixelArr = []
    for pixel in pixelPos:
        pixelArr.append([pixelVal[pixel], pixelVal[pixel], pixelVal[pixel]])
    return pixelArr

def get_pixel_pos(n):
    arr = []
    for i in range(0, n):
        arr.append(i)
    return arr

# gets the Gamma for one pixel, refPos is pixel
def get_1D_gamma_full_for_one_pixel(refPos):
    # refPos is from original image

    # define edges of the search based on the search_size window
    # units are in pixel index
    left_edge = 0 if refPos * spacing - search_size <= 0 else (refPos * spacing - search_size) 
    right_edge = (len(image2)) * spacing if refPos * spacing + search_size > len(image2) * spacing else refPos * spacing + search_size

    # imageList is based on image2 (the image we are comparing to)
    imageList = image2
    imageList = test_image_pos_to_interp_data(image2)
    # x vals
    imageList[0] = image_data_to_real_units([0, *imageList[0]])
    # y vals
    imageList[1] = [image2[0], *imageList[1]]

    lin_val_interp = interp1d(imageList[0], imageList[1])
    # pixel on the original image we will be compare to
    refVal = image1[refPos]

    # forwards search, include the pos itself since it might be different on the test image (image2)
    i = refPos*spacing
    gamma = []
    pos = []

    while (i <= right_edge):
        currVal = lin_val_interp(i)
        # if currGamma <= 1, pass
        # get reference from the ground truth image, the currVal is from the image you want to test
        currGamma = math.sqrt((((refPos*spacing - i) ** 2) / search_size) + ((refVal - currVal) ** 2) / search_percent)
        gamma.append(currGamma)
        pos.append(i)
        i += step_size

    # backwards search
    i = refPos*spacing - step_size
    while (i >= left_edge):
        currVal = lin_val_interp(i)
        # if currGamma <= 1, pass
        currGamma = math.sqrt((((refPos*spacing - i) ** 2) / search_size) + ((refVal - currVal) ** 2) / search_percent)
        gamma = [currGamma, *gamma]
        pos = [i, *pos]
        i -= step_size

    return gamma

def get_passing_rate():
    totalPass = 0
    for i in range(0, len(image1)):
        currGammaList = get_1D_gamma_full_for_one_pixel(i)
        currGamma = min(currGammaList)
        if (currGamma <= 1):
            totalPass += 1
    passDecimal = totalPass/len(image1)

    return str(passDecimal * 100) + '%'

def get_gamma_image():
    # refPos is from reference image, original
    # testing if test image is similar to original, low gamma = same
    # for every pixel on the original image, get the gamma value from doing algorithm on the test image
    for i in range(0, len(image1)):
        currGammaList = get_1D_gamma_full_for_one_pixel(i)
        currGamma = min(currGammaList)
        print(str(i) + " - Gamma: " + str(currGamma))
        gammaImage.append(currGamma)
    imageArr = arrays_to_img(gammaImage, get_pixel_pos(len(gammaImage)))
    npImageArr = np.array(imageArr)
    return npImageArr

def main():
    
    plt.figure("Original Reference Image")
    imageArr1 = arrays_to_img(image1, get_pixel_pos(len(image1)))
    npImageArr1 = np.array(imageArr1)
    imgOriginal = plt.imshow(npImageArr1, cmap='gray')

    plt.figure("Test Image")
    imageArr2 = arrays_to_img(image2, get_pixel_pos(len(image2)))
    npImageArr2 = np.array(imageArr2)
    imgTest = plt.imshow(npImageArr2, cmap='gray')

    plt.figure("Gamma Image")
    npGammaArr = get_gamma_image()
    gammaImg = plt.imshow(npGammaArr, cmap='gray')
    passingRate = get_passing_rate()
    plt.text(-5, 0, 'Passing Rate: ' + str(passingRate), bbox=dict(fill=False, edgecolor='red', linewidth=3))

    plt.show()
    

if (__name__ == '__main__'):
    main()