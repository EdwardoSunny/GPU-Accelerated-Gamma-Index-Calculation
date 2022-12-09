#!/usr/bin/env python3

import math
from numba import cuda
from numba import vectorize
import numpy as np

@cuda.jit(device=True)
def get_interp_image_x_y(xRange, yRange):
    xData = np.zeros(xRange)
    yData = np.zeros(yRange)
    for i in range (1, xRange+1):
        xData[i-1] = i
    for i in range(1, yRange+1):
        yData[i-1] = i
    return [xData, yData]
