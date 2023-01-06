#!/usr/bin/env python3
import numpy as np
import time

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
  # Calculate the gamma index
  gamma = np.abs(dose_planned - dose_actual) / dose_threshold
  gamma = np.where(gamma <= 1, gamma, np.inf)
  gamma = np.where(dose_planned > 0, gamma, 0)

  # Calculate the distance-to-agreement
  dta = np.sqrt((np.square(np.where(dose_planned > dose_actual, dose_planned, dose_actual))) / (dose_threshold ** 2))
  dta = np.where(dta <= dta_threshold, dta, np.inf)

  # Combine the gamma index and distance-to-agreement
  gamma = np.where(gamma > dta, gamma, dta)

  return gamma

dose_planned = np.random.rand(1000, 1000).astype(np.float32)
dose_actual = np.random.rand(1000, 1000).astype(np.float32)
dose_threshold = 1.0
dta_threshold = 1.0
index = gamma_index(dose_planned, dose_actual, dose_threshold, dta_threshold)

st = time.time()
gamma_index = gamma_index(dose_planned, dose_actual, dose_threshold, dta_threshold)
et = time.time()
print(gamma_index)
print("\n" + "Time Spent: " + str(et-st))
