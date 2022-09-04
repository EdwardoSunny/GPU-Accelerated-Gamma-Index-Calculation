from scipy import interpolate

X = [0, 2, 3, 4, 5, 6, 7, 8]

Y = [1,2,3,6,5,6,7,8] # random y values

Z = [5,2,3,6,5,6,7,8]



interpFunction = interpolate.interp2d(X, Y, Z, kind='linear')

val = interpFunction(1000, 1000)
print(val)
