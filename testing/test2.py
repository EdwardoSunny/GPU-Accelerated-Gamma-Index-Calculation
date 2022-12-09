#!/usr/bin/env python3
from numba import cuda
from numba import *

# f8 is float
# https://www.geeksforgeeks.org/data-type-object-dtype-numpy-python/
test_gpu = cuda.jit(uint32, device=True)(test)

@cuda.jit(argtypes=[uint32])
def test(n):
    for i in range(0, n):
        for j in range (0, n):
            n += 1
