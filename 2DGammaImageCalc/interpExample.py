from fast_interp import interp2d
import numpy as np

nx = 50
ny = 37
xv, xh = np.linspace(0, 1,       nx, endpoint=True,  retstep=True)
yv, yh = np.linspace(0, 2*np.pi, ny, endpoint=False, retstep=True)
x, y = np.meshgrid(xv, yv, indexing='ij')

test_function = lambda x, y: np.exp(x)*np.exp(np.sin(y))
f = test_function(x, y)
print(type(f))
print(str(len(f)) + " " + str(len(f[0])))
test_x = -xh/2.0
test_y = 271.43
fa = test_function(test_x, test_y)

interpolater = interp2d([0,0], [1,2*np.pi], [xh,yh], f, k=5, p=[False,True], e=[1,0])
fe = interpolater(test_x, test_y)

print('Error is: {:0.2e}'.format(np.abs(fe-fa)))
