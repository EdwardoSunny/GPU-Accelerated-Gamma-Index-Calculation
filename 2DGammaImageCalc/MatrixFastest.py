#!/usr/bin/env python3
import numba
from numba import cuda
import numpy as np
import math

@cuda.jit
def gamma_index_kernel(dose_planned, dose_actual, dose_threshold, dta_threshold, gamma):
  """
  GPU kernel for calculating the gamma index.
  """
  # Calculate the gamma index at this point in the image
  gamma_val = abs(dose_planned - dose_actual) / dose_threshold
  gamma_val = gamma_val if gamma_val <= 1 else np.inf
  gamma_val = gamma_val if dose_planned > 0 else 0

  # Calculate the distance-to-agreement at this point in the image
  dta = math.sqrt(math.pow(max(dose_planned, dose_actual), 2) / math.pow(dose_threshold, 2))
  dta = dta if dta <= dta_threshold else np.inf

  # Combine the gamma index and distance-to-agreement
  gamma = gamma_val if gamma_val > dta else dta

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
  # Convert the input arrays to float32 to match the kernel function signature
  dose_planned = dose_planned.astype(np.float32)
  dose_actual = dose_actual.astype(np.float32)

  # Create an output array to store the gamma index map
  gamma = np.empty_like(dose_planned, dtype=np.float32)

  # Call the GPU kernel
  gamma_index_kernel[1, (dose_planned.size + 255) // 256](dose_planned, dose_actual, dose_threshold, dta_threshold, gamma)

  return gamma

dose_planned = np.random.rand(10, 10).astype(np.float32)
dose_actual = np.random.rand(10, 10).astype(np.float32)
dose_threshold = 1.0
dta_threshold = 1.0
index = gamma_index(dose_planned, dose_actual, dose_threshold, dta_threshold)

st = time.time()
gamma_index = gamma_index(dose_planned, dose_actual, dose_threshold, dta_threshold)
et = time.time()
print(gamma_index)
print("\n" + "Time Spent: " + str(et-st))
