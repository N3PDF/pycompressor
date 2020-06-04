
"""
Tests for the estimators.py module
"""

import numpy as np
from pycompressor import estimators
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal

# Define test values
FLAVS = 3
XGRID = 6
PDFSIZE = 10

# Create Toy PDFprior = np.random.rand(total, FLAVS, XGRID)
REPLICAS = np.random.rand(PDFSIZE, FLAVS, XGRID)

# Initialize estimator class
estimator = estimators.Estimators(REPLICAS)


def _mean(replicas):
    """ Compute meanin the standard way. """
    result = np.zeros((FLAVS, XGRID))
    for fl in range(FLAVS):
        for x in range(XGRID):
            repl = 0
            for rep in range(PDFSIZE):
                repl += replicas[rep][fl][x]
            result[fl][x] = repl
    return result / PDFSIZE


def _stdev(replicas):
    """ Compute stdev in the standard way. """
    fac = PDFSIZE - 1
    result = np.zeros((FLAVS, XGRID))
    for fl in range(FLAVS):
        for x in range(XGRID):
            su_repl, sq_repl = 0, 0
            for rep in range(PDFSIZE):
                repl = replicas[rep][fl][x]
                su_repl += repl
                sq_repl += repl * repl
            a = sq_repl / fac
            b = PDFSIZE / fac * pow(su_repl / PDFSIZE, 2)
            result[fl][x] = np.sqrt(a - b)
    return result


def test_mean():
    """ Compare standard mean and the implemented
    mean in the estimator.py module. """
    py_mean = estimator.compute_for('mean')
    st_mean = _mean(REPLICAS)
    assert_array_equal(py_mean, st_mean)


def test_stdev():
    """ Compare standard stdev and the implemented
    stdev in the estimator.py module. """
    py_stdev = estimator.compute_for('stdev')
    st_stdev = _stdev(REPLICAS)
    assert_almost_equal(py_stdev, st_stdev, decimal=10)


# The above also tests the validity of the other moment
# estimators as the computations are done excatly in the
# same way and they output excatly the same array shape.


# def _ks_c_compressor(mean, stdv, flavour, x, nb_regions):
#     res = np.zeros(nb_regions)
#     for r in range(PDFSIZE):
#         val = REPLICAS[r][flavour][x]
#         if val <= mean[flavour][x] - 2 * stdv[flavour][x]:
#             res[0] += 1
#         if val <= mean[flavour][x] - 1 * stdv[flavour][x]:
#             res[1] += 1
#         if val <= mean[flavour][x]:
#             res[2] += 1
#         if val <= mean[flavour][x] + 1 * stdv[flavour][x]:
#             res[3] += 1
#         if val <= mean[flavour][x] + 2 * stdv[flavour][x]:
#             res[4] += 1
#         if val > mean[flavour][x] + 2 * stdv[flavour][x]:
#             res[5] += 1
#     return res / PDFSIZE
# 
# 
# def test_kolmogorov():
#     """ Compute Kolmogorov-smirnov (KS) estimator as in the C-implementation
#     of the compressor:
# 
#     https://github.com/scarrazza/compressor/blob/master/src/Estimators.cc#L122
# 
#     As opposed to the above implementation, this computes the KS for all
#     replicas, flavours and x-grid. """
#     nb_regions = 6
#     # mean = _mean(REPLICAS)
#     # stdv = _stdev(REPLICAS)
#     mean = estimator.compute_for('mean')
#     stdv = estimator.compute_for('stdev')
# 
#     st_ks = np.zeros((FLAVS, XGRID, nb_regions))
#     for fl in range(FLAVS):
#         for x in range(XGRID):
#             st_ks[fl][x] = _ks_c_compressor(mean, stdv, fl, x, nb_regions)
# 
#     # KS computed from pyCompressor
#     py_ks = estimator.compute_for('kolmogorov_smirnov')
#     assert_almost_equal(py_ks, st_ks, decimal=10)


def test_kolmogorov():
    """ Compute Kolmogorov-smirnov (KS) estimator as in the C-implementation
    of the compressor:

    https://github.com/scarrazza/compressor/blob/master/src/Estimators.cc#L122

    As opposed to the above implementation, this computes the KS for all
    replicas, flavours and x-grid. """
    nb_regions = 6
    mean = estimator.compute_for('mean')
    stdv = estimator.compute_for('stdev')

    def _ks_c_compressor(flavour, x):
        res = np.zeros(nb_regions)
        for r in range(PDFSIZE):
            val = REPLICAS[r][flavour][x]
            if val <= mean[flavour][x] - 2 * stdv[flavour][x]:
                res[0] += 1
            elif val <= mean[flavour][x] - 1 * stdv[flavour][x]:
                res[1] += 1
            elif val <= mean[flavour][x]:
                res[2] += 1
            elif val <= mean[flavour][x] + 1 * stdv[flavour][x]:
                res[3] += 1
            elif val <= mean[flavour][x] + 2 * stdv[flavour][x]:
                res[4] += 1
            elif val > mean[flavour][x] + 2 * stdv[flavour][x]:
                res[5] += 1
        return res / PDFSIZE
    st_ks = np.zeros((FLAVS, XGRID, nb_regions))
    for fl in range(FLAVS):
        for x in range(XGRID):
            st_ks[fl][x] = _ks_c_compressor(fl, x)

    # KS computed from pyCompressor
    py_ks = estimator.compute_for('kolmogorov_smirnov')
    assert_almost_equal(py_ks, st_ks, decimal=10)
