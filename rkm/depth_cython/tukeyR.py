"""Compute the tukey depth adapted from the R implementation.
    https://rdrr.io/cran/DepthProc/src/R/depth.R

References (https://rdrr.io/cran/DepthProc/man/depth.html)

	1. Liu, R.Y., Parelius, J.M. and Singh, K. (1999), Multivariate analysis by data depth: Descriptive statistics, graphics and inference (with discussion), Ann. Statist., 27, 783–858.
	2. Mosler K (2013). Depth statistics. In C Becker, R Fried, K S (eds.), Robustness and Complex Data Structures, Festschrift in Honour of Ursula Gather, pp. 17–34. Springer.
	3. Rousseeuw, P.J. and Struyf, A. (1998), Computing location depth and regression depth in higher dimensions, Stat. Comput., 8, 193–203.
	4. Zuo, Y. and Serfling, R. (2000), General Notions of Statistical Depth Functions, Ann. Statist., 28, no. 2, 461–482.


"""
import copy
import os.path

import numpy as np
from statsmodels.distributions import ECDF


def depthTukeyCPP(U, X, exact=True, threads=-1):
	"""

	https://github.com/zzawadz/DepthProc/blob/master/src/Depth.cpp

	Parameters
	----------
	U
	X
	exact
	threads

	Returns
	-------

	"""
	# if threads < 1: threads = omp_get_max_threads()

	import ctypes
	from ctypes import cdll
	import glob
	# load the library
	# find the shared library, the path depends on the platform and Python version
	lib_dir = os.path.abspath('.')
	print(lib_dir)
	lib_file = glob.glob(f'{lib_dir}/**/*TukeyDepth*.so', recursive=True)[0]
	# lib = cdll.LoadLibrary('libTukeyDepth.so')
	lib = cdll.LoadLibrary(lib_file)

	n, d = X.shape
	depths = np.zeros((n, ))
	for i in range(n):
		# data = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
		data = X
		nrows, ncols = data.shape
		x = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		lib.depthTukey2dExact_PY.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
		                                     ctypes.POINTER(ctypes.c_double)]
		lib.depthTukey2dExact_PY.restype = ctypes.c_double
		# f = lib.depthTukey2dExact_PY(ctypes.c_double(1), ctypes.c_double(2.0), ctypes.c_int(n), ctypes.c_int(m), z)
		depths[i] = lib.depthTukey2dExact_PY(U[i][0], U[i][1], nrows, ncols, x)

	return depths


def runifsphere(npoints, ndim=3, random_state=42):
	""" Random number generation from unit sphere.

	CPP source code:
	https://github.com/zzawadz/DepthProc/blob/7d676879a34d49416fb00885526e27bcea119bbf/src/Utils.cpp

	arma::mat runifsphere(size_t n, size_t p)
	{
		//arma::mat X(n, p);
		# https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Normal.html: rnorm(n, mean = 0, sd = 1)
		NumericVector rx = rnorm(n*p);
        arma::mat X(rx.begin(), n, p, false);   # reshape rx to (n, p)
        //X.randn();
		arma::vec norm = arma::sum(X % X, 1);
		norm = arma::sqrt(norm);
		X.each_col() /= norm;
		return X;
	}

	Returns
	-------

	"""
	# https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
	rng = np.random.RandomState(seed=random_state)
	vec = rng.randn(npoints, ndim)  # Return a sample (or samples) from the "standard normal" distribution.
	vec /= np.linalg.norm(vec, axis=0)
	return vec


def tukey1d(u, X):
	"""
	Rousseeuw, P.J. and Struyf, A. (1998), Computing location depth and regression depth in higher dimensions, Stat. Comput., 8, 193–203.
	when p = 1.

	Parameters
	----------
	u: nx1
	X: nx1

	Returns
	-------

	"""
	Xecdf = ECDF(X)  # Return the Empirical CDF of an array (X) as a step function.
	uecdf = Xecdf(u)  # number of points of X in the left side ( x <= u) of the hyperplane (with boundary through u)
	uecdf2 = 1 - uecdf  # number of points of X in the right side (x => u) of the hyperplane (with boundary through u)
	min_ecdf = uecdf > uecdf2  # indices of points that left > right
	depth = uecdf
	depth[min_ecdf] = uecdf2[min_ecdf]
	return depth


def depth_tukey(u=None, X=None, ndir=1000, threads=-1, exact=False, random_state=42):
	"""
	u: the point(s) (i.e., reference point) that you want the tukey depth
	X: the given dataset
	ndir: number of directions, i.e., number of hyperplanes for each reference point of u.

	@title Tukey Depth
	@export
	@description Computes the Tukey depth of a point or vectors of points with respect to a multivariate data set.

	@param u Numerical vector or matrix whose depth is to be calculated. Dimension has to be the same as that of the observations.
	@param X The data as a matrix, data frame or list. If it is a matrix or data frame, then each row is viewed as one multivariate observation. If it is a list, all components must be numerical vectors of equal length (coordinates of observations).
	@param ndir number of directions used in computations
	@param threads number of threads used in parallel computations. Default value -1 means that all possible cores will be used.
	@param exact if TRUE exact alhorithm will be used . Currently it works only for 2 dimensional data set.

	@details

	Irrespective of dimension, Projection and Tukey's depth is obtained by approximate calculation.

	Returns the depth of multivariate point \code{u} with respect to data set \code{X}.

	@author Daniel Kosiorowski, Mateusz Bocian, Anna Wegrzynkiewicz and Zygmunt Zawadzki from Cracow University of Economics.

	@examples
	\dontrun{
	x <- matrix(rnorm(3000), nc = 3)
	depthTukey(x, ndir = 2000)
	}

	# Exact algorithm in 2d
	x <- matrix(rnorm(2000), nc = 2)
	depthTukey(x, exact = TRUE)

	@keywords
	multivariate
	nonparametric
	depth function

	References:
		Rousseeuw, P.J. and Struyf, A. (1998), Computing location depth and regression depth in higher dimensions, Stat. Comput., 8, 193–203.

	"""


	n, d = X.shape
	if u is None:
		# Create a matrix u from the given data X
		u = np.array(X, copy=True)

	if d == 1:
		# Rousseeuw, P.J. and Struyf, A. (1998), Computing location depth and regression depth in higher dimensions, Stat. Comput., 8, 193–203.
		depth = tukey1d(u, X)
	elif (d == 2 and exact):
		# return depthTukeyCPP(u, X, exact, threads)
		# https://github.com/zzawadz/DepthProc/blob/master/src/Depth.cpp
		# https://github.com/zzawadz/DepthProc/blob/7d676879a34d49416fb00885526e27bcea119bbf/src/TukeyDepth.cpp
		# raise NotImplemented('Not implemented yet.')
		depth = depthTukeyCPP(u, X, exact, threads)
	else:
		# Approximate calculation of Tukey depth
		# if number of dimensions is greater than 2
		proj = runifsphere(ndir, ndim=d, random_state=random_state).transpose()  # shape: (nxd)^T
		# each vector/basis (u) has unit norm, i.e., ||u|| = 1
		# %*%:  https://www.programmingr.com/matrix-multiplication/
		xut = X @ proj  # xut: nxn Project each point x in X onto span(proj) space.
		uut = u @ proj  # uut: nxn Project each reference point u onto span(proj) space.

		OD = np.zeros(uut.shape)

		for j in range(0, ndir):
			# For each basis of span(proj) (where we project all points(X) onto this basis(i.e., this direction);
			# then compute Tukey depth for this direction
			# OD[, i] <- tukey1d(uut[, i], xut[, i])
			OD[:, j] = tukey1d(uut[:, j], xut[:, j])
		depth = np.min(OD, axis=1)

	return depth


def tukey_median(X, exact=False, random_state=42):
	u = copy.deepcopy(X)
	depths = depth_tukey(u=u, X=X, ndir=1000, threads=-1, exact=exact, random_state=random_state)
	max_index = np.argmax(depths)
	return u[max_index], depths[max_index], max_index


if __name__ == '__main__':
	random_state = 42
	rng = np.random.RandomState(seed=random_state)
	# u = rng.multivariate_normal(np.array([0, 0]), np.diag([1, 1]), 10)  # Reference points (can be any positions)
	X = rng.multivariate_normal(np.array([0, 0]), np.diag([1, 1]), 100)  # Dataset X
	res = depth_tukey(u=None, X=X, random_state=random_state)
	print(res.shape, res)
	med, depth, index = tukey_median(X, exact=True, random_state=random_state)
	print(med, depth, index)
