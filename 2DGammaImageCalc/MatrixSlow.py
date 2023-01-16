#!/usr/bin/env python3
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numba
import numpy as np
import time
import cv2

@numba.vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def gamma_index_gpu(dose_planned, dose_actual, dose_threshold, dta_threshold):
  """
  GPU-accelerated function for calculating the gamma index.
  """
  # Calculate the gamma index at this point in the image
  gamma_val = abs(dose_planned - dose_actual) / dose_threshold
  gamma_val = np.clip(gamma_val, 0, 1)
  gamma_val = gamma_val if dose_planned > 0 else 0

  # Calculate the distance-to-agreement at this point in the image
  dta = np.sqrt(max(dose_planned, dose_actual)**2 / dose_threshold**2)
  dta = np.clip(dta, 0, dta_threshold)

  # Combine the gamma index and distance-to-agreement
  return gamma_val if gamma_val > dta else dta


def gamma_index(dose_planned, dose_actual, dose_threshold, dta_threshold):
  """
  Calculate the gamma index for a given 2D image.

  Parameters:
  - dose_planned: 2D numpy array, the planned dose distribution
  - dose_actual: 2D numpy array, the actual dose distribution
  - dose_threshold: float, the dose difference threshold
  - dta_threshold: float, the distance-to-agreement threshold

  Returns:
  - gamma: 2D numpy array, the gamma index map
  """
  # Convert the input arrays to float32 to match the function signature
  dose_planned = dose_planned.astype(np.float32)
  dose_actual = dose_actual.astype(np.float32)

  # Call the GPU-accelerated function
  gamma = gamma_index_gpu(dose_planned, dose_actual, dose_threshold, dta_threshold)

  return gamma

dose_actual = cv2.imread('ref.png').astype(np.float32) # reference image
dose_planned = cv2.imread('test.png').astype(np.float32) # test image
plt.figure("reference")
dose_actual_image = plt.imshow(dose_actual, cmap='gray')
plt.figure("planned")
dose_planned_image = plt.imshow(dose_planned, cmap='gray')

#dose_planned = np.random.rand(10, 10).astype(np.float32)
#dose_actual = np.random.rand(10, 10).astype(np.float32)
dose_threshold = 0.1
dta_threshold = 0.1
index = gamma_index(dose_planned, dose_actual, dose_threshold, dta_threshold)

st = time.time()
gamma_index = gamma_index(dose_planned, dose_actual, dose_threshold, dta_threshold)
et = time.time()
print(gamma_index)
plt.figure("gamma")
gamma_map = plt.imshow(gamma_index, cmap='gray')
plt.show()
print("\n" + "Time Spent: " + str(et-st))

# Output: [[0 0 0]
#          [0 1 0]
#          [0 0 0]]
