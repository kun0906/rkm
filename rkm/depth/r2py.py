"""
	Convert R scripts 2 Python scripts by pyensae

	pip install pyensae
	http://www.xavierdupre.fr/app/pyensae/helpsphinx/notebooks/r2python.html

"""
rscript = """
depthTukey <- function(u, X, ndir = 1000, threads = -1, exact = FALSE) {

  if (missing(X)) {
    X <- u
  }

  if (is.vector(u)) u <- matrix(u, ncol = ncol(X))

  tukey1d <- function(u, X) {
    Xecdf <- ecdf(X)
    uecdf <- Xecdf(u)
    uecdf2 <- 1 - uecdf
    min.ecdf <- uecdf > uecdf2
    depth <- uecdf
    depth[min.ecdf] <- uecdf2[min.ecdf]
    depth
  }

  if (ncol(X) == 1) {
    depth <- tukey1d(u, X)
  } else if (ncol(X) == 2 && exact) {
    depth <- depthTukeyCPP(u, X, exact, threads)
  } else {
    # if number of dimensions is greater than 2
    proj <- t(runifsphere(ndir, ncol(X)))
    xut <- X %*% proj
    uut <- u %*% proj

    OD <- matrix(nrow = nrow(uut), ncol = ncol(uut))

    for (i in 1:ndir) {
      OD[, i] <- tukey1d(uut[, i], xut[, i])
    }

    depth <- apply(OD, 1, min)
  }

  new("DepthTukey", depth, u = u, X = X, method = "Tukey")
}
"""

# rscript = """
# nb=function(y=1930){
# debut=1816
# MatDFemale=matrix(D$Female,nrow=111)
# colnames(MatDFemale)=(debut+0):198
# cly=(y-debut+1):111
# deces=diag(MatDFemale[:,cly[cly%in%1:199]])
# return(c(B$Female[B$Year==y],deces))}
# """

from pyensae.languages.rconverter import r2python
print(r2python(rscript, pep8=True))
