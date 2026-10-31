"""Package-wide defaults for fitting Gaussian models of any covariance structure."""

DEFAULT_RIDGE = 1e-6  # package-wide default for relative ridge regularization

# package-wide default for relative covariance floor regularization at fitting
DEFAULT_COV_FLOOR = 1e-6

# package-wide default for relative covariance floor regularization at
#  initialization, which is more conservative than the default for fitting
DEFAULT_COV_FLOOR_INIT = 1e-2
