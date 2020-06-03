"""
Tests for the err_function.py module
"""

import numpy as np
from pycompressor import err_function

# Define test values
FLAVS = 3
XGRID = 6
PDFSIZE = 10

# Create Toy PDFprior = np.random.rand(total, FLAVS, XGRID)
PRIOR = np.random.rand(PDFSIZE, FLAVS, XGRID)

# List of estimators
ESTIMATORS = {
        'moment_estimators': [
            'mean',
            'stdev',
            'skewness',
            'kurtosis',
            'moment5th',
            'moment6th'
            ],
        'stat_estimators': [
            'kolmogorov_smirnov'
            ]
        }


def get_subset(n):
    """ Extract random set of replicas from the prior
    using the `randomize_rep` method. """
    subset = err_function.randomize_rep(PRIOR, n)
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

    IMPROTANT:
    =========
    This at the same time tests the Estimators. """
    est = err_function.estimate(PRIOR, ESTIMATORS)
    for key, val in est.items():
        if key in ESTIMATORS['moment_estimators']:
            assert est[key].shape == (PRIOR.shape[1], PRIOR.shape[2])
        if key in ESTIMATORS['stat_estimators']:
            assert est[key].shape == (PRIOR.shape[1], PRIOR.shape[2], 6)
    return est


def test_normalization(random_size=4, trial=2):
    """ Test if the value of the normalization factors
    are positive. """
    est_prior = test_estimate()
    norm = err_function.normalization(
                PRIOR,
                est_prior,
                random_size,
                ESTIMATORS,
                trial
            )
    for _, val in norm.items():
        assert val > 0
