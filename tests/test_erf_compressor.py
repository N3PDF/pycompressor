# Tests for the errfunction.py module

import os
import pytest
import numpy as np
from numpy.random import PCG64
from numpy.random import Generator
from pycompressor import compressor
from pycompressor import errfunction

# Define test values
FLAVS = 3
# The x-grid has to be big enough in order for the
# correlation matrix not to be SINGULAR
XGRID = 20
PDFSIZE = 10
NB_REDUCED = 4

# Seed
RDM = Generator(PCG64(seed=0))

# Create Toy prior PDF
PRIOR = np.random.uniform(0, 1, size=[PDFSIZE, FLAVS, XGRID])

# List of estimators
ESTIMATORS = {
    "moment_estimators": ["mean", "stdev", "skewness", "kurtosis"],
    "stat_estimators": ["kolmogorov_smirnov"],
    "corr_estimators": ["correlation"],
}

# Create test folder
# Create Output folders
if not os.path.exists("TEST"):
    os.mkdir("TEST")
else:
    pass


# Compressor class
COMP = compressor.compress(
    PRIOR,
    ESTIMATORS,
    NB_REDUCED,
    "TEST",
    RDM
)


def get_subset(n):
    """ Extract random set of replicas from the prior
    using the `randomize_rep` method. """
    subset = errfunction.randomize_rep(PRIOR, n, RDM)
    return subset


def test_randomize_rep(n=4):
    """ Ensures that the output of the function
    is indeed a subset of the input """
    subset = get_subset(n)
    for replica in subset:
        assert replica in PRIOR


def test_estimate():
    """ Test the computation of the ERF. This amounts to
    checking that the outputs of the method `estimate` give
    the right shape.

    IMPORTANT:
    =========
    This at the same time tests the Estimators. """
    est = errfunction.estimate(PRIOR, ESTIMATORS)
    for key, val in est.items():
        if key in ESTIMATORS["moment_estimators"]:
            moment_est_shape = (PRIOR.shape[1], PRIOR.shape[2])
            assert est[key].shape == moment_est_shape
        if key in ESTIMATORS["stat_estimators"]:
            stat_est_shape = (PRIOR.shape[1], PRIOR.shape[2], 6)
            assert est[key].shape == stat_est_shape
    return est


def test_normalization(random_size=4, trial=2):
    """ Test if the value of the normalization factors
    are positive. """
    est_prior = test_estimate()
    norm = errfunction.normalization(
        PRIOR,
        est_prior,
        random_size,
        ESTIMATORS,
        trial,
        "TEST",
        RDM
    )
    for _, val in norm.items():
        assert val > 0


def test_complete_erf():
    """ Test ERF computation by selecting randomly a subset of
    random replicas. """
    index = np.random.randint(PDFSIZE, size=NB_REDUCED)
    erf_random = COMP.error_function(index)
    assert erf_random > 0 and erf_random < 4


def test_ga():
    """ Test that the Genetic Algorithm deos what it is supposed
    to be--meaning--it reduces the ERF value at each iteration."""
    ref_erf = 5
    for _ in range(4):
        best_erf, _ = COMP.genetic_algorithm()
        assert best_erf > 0 and best_erf <= ref_erf
        ref_erf = best_erf


@pytest.mark.skip(reason="CMA not working properly yet")
def test_cma():
    """
    Test that CMA minimizes the Error Function.
    """
    seed = 0
    verbose = 0
    init_standard_dev = 0.3
    sel1_erf, _ = COMP.cma_algorithm(
        std_dev=init_standard_dev,
        verbosity=verbose,
        min_itereval=1,
        seed=seed,
        max_itereval=2,
    )
    sel2_erf, _ = COMP.cma_algorithm(
        std_dev=init_standard_dev,
        verbosity=verbose,
        min_itereval=4,
        seed=seed,
        max_itereval=10,
    )
    assert sel2_erf <= sel1_erf
