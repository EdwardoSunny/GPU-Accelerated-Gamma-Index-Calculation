import time
import numpy as np
from numba import jit, njit, vectorize

def og_function(input_list):
    # would be better if this was preallocated by making a zeros np array and then setting it
    output_list = []
    for item in input_list:
        if item % 2:
            output_list.append(2)
        else:
            output_list.append(1)
    return output_list

st = time.time()
test_arr = np.arange(1000000)
og_function(test_arr)
et = time.time()

print("og " + str(et-st))
jitted_function = njit()(og_function)

st = time.time()
jitted_function(test_arr)

et = time.time()

print("jitted compile " + str(et-st))
st = time.time()
jitted_function(test_arr)

et = time.time()

print("jitted " + str(et-st))

# runs on per element, can use per element but also can pass in a list and it will apply the function to every single element
@vectorize
def scalar_comp(num):
    if num % 2:
        return 2
    else:
        return 1

st = time.time()
scalar_comp(test_arr)

et = time.time()
print("vectorize compile " + str(et-st))

st = time.time()
scalar_comp(test_arr)

et = time.time()
print("vectorize " + str(et-st))
