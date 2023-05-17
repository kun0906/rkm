"""

"""

# import the module
import numpy as np
cimport numpy as np

cdef extern from "TukeyDepth.h":
	double depthTukey2dExact_PY(double U, double V, int nrows, int ncols, double *A)



cdef depthTukey2dExact_np(U, V,nrows, ncols,
                          np.ndarray[double, ndim=2, mode="c"] A,
                          ):

	return depthTukey2dExact_PY(U, V, nrows, ncols,
	                         <double *> np.PyArray_DATA(A)
	                         )
#
#
# class Depth:
#
# 	def __init__(self):
# 		pass
#
# 	def depthTukey2dExact_PY2(self, U, V, nrows, ncols, A):
# 		return depthTukey2dExact_np(U, V, nrows, ncols, A)
