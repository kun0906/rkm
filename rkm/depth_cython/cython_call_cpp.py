"""

Ctypes:
	https://www.geeksforgeeks.org/how-to-call-c-c-from-python/
	https://github.com/dvida/py-ctypes-multidimensional-arrays/blob/master/myfunc.py


"""

# import the module
import numpy as np
import ctypes
from ctypes import cdll
import glob

# load the library
# find the shared library, the path depends on the platform and Python version
lib_file = glob.glob('build/*/TukeyDepth2*.so')[0]
# lib = cdll.LoadLibrary('libTukeyDepth.so')
lib = cdll.LoadLibrary(lib_file)

data = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
z = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
lib.depthTukey2dExact_PY.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.depthTukey2dExact_PY.restype=ctypes.c_double
nrows, ncols = data.shape
# f = lib.depthTukey2dExact_PY(ctypes.c_double(1), ctypes.c_double(2.0), ctypes.c_int(n), ctypes.c_int(m), z)
f = lib.depthTukey2dExact_np(1, 2.0, nrows, ncols, z)
print(f"f:{f}")



