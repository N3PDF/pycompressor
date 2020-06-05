"""
Tests for the estimators.py module
"""

import numpy as np
from pycompressor import estimators
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal

# Define test values
FLAVS = 3
# The x-grid has to be big enough in order for the
# correlation matrix not to be SINGULAR
XGRID = 20
PDFSIZE = 10
FAC = PDFSIZE - 1

# Seed
np.random.seed(0)

# Create Toy prior PDF
REPLICAS = np.random.uniform(0, 1, size=[PDFSIZE, FLAVS, XGRID])

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
    result = np.zeros((FLAVS, XGRID))
    for fl in range(FLAVS):
        for x in range(XGRID):
            su_repl, sq_repl = 0, 0
            for rep in range(PDFSIZE):
                repl = replicas[rep][fl][x]
                su_repl += repl
                sq_repl += repl * repl
            a = sq_repl / FAC
            b = PDFSIZE / FAC * pow(su_repl / PDFSIZE, 2)
            result[fl][x] = np.sqrt(a - b)
    return result


# Compute mean and stdev for
# for future calls
MEAN = _mean(REPLICAS)
STDV = _stdev(REPLICAS)


def _skewness(replicas):
    """ Compute skewness in the standard way following
    exactly eq.(11) of the paper. """
    result = np.zeros((FLAVS, XGRID))
    for fl in range(FLAVS):
        for x in range(XGRID):
            suma = 0
            for rep in range(PDFSIZE):
                repl = replicas[rep][fl][x] - MEAN[fl][x]
                suma += pow(repl, 3)
            result[fl][x] = suma / pow(STDV[fl][x], 3)
    return result / PDFSIZE


# Here are the tests of estimators.py modules
# The results below are compared with the formal
# definitions above


def test_mean():
    """ Compare standard mean and the implemented
    mean in the estimator.py module. """
    py_mean = estimator.compute_for("mean")
    assert_array_equal(py_mean, MEAN)


def test_stdev():
    """ Compare standard stdev and the implemented
    stdev in the estimator.py module. """
    py_stdev = estimator.compute_for("stdev")
    assert_almost_equal(py_stdev, STDV, decimal=10)


def test_skewness():
    """ Compare the skewness implementation to the
    standard one. """
    st_skw = _skewness(REPLICAS)
    py_skw = estimator.compute_for('skewness')
    assert_almost_equal(py_skw, st_skw, decimal=10)


def test_kolmogorov():
    """ Compute Kolmogorov-smirnov (KS) estimator as in the C-implementation
    of the compressor:

    https://github.com/scarrazza/compressor/blob/master/src/Estimators.cc#L122

    As opposed to the above implementation, this computes the KS for all
    replicas, flavours and x-grid. """
    nb_regions = 6

    def _ks_c_compressor(flavour, x):
        res = np.zeros(nb_regions)
        for r in range(PDFSIZE):
            val = REPLICAS[r][flavour][x]
            if val <= MEAN[flavour][x] - 2 * STDV[flavour][x]:
                res[0] += 1
            elif val <= MEAN[flavour][x] - 1 * STDV[flavour][x]:
                res[1] += 1
            elif val <= MEAN[flavour][x]:
                res[2] += 1
            elif val <= MEAN[flavour][x] + 1 * STDV[flavour][x]:
                res[3] += 1
            elif val <= MEAN[flavour][x] + 2 * STDV[flavour][x]:
                res[4] += 1
            elif val > MEAN[flavour][x] + 2 * STDV[flavour][x]:
                res[5] += 1
        return res / PDFSIZE

    st_ks = np.zeros((FLAVS, XGRID, nb_regions))
    for fl in range(FLAVS):
        for x in range(XGRID):
            st_ks[fl][x] = _ks_c_compressor(fl, x)

    # KS computed from pyCompressor
    py_ks = estimator.compute_for("kolmogorov_smirnov")
    assert_almost_equal(py_ks, st_ks, decimal=10)


def test_correlation():
    """ This function test that correlation matrix computed
    from the correlation estimator is INVERTIBLE (or non-
    singular). """
    corr_est = estimator.compute_for("correlation")
    np.linalg.inv(corr_est)
