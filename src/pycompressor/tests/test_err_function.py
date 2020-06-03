"""
Tests for the err_function.py module
"""
from scipy import stats
import numpy as np
from pycompressor import err_function

FLAVS = 3
XGRID = 10

def get_subset(n, total):
    prior = np.random.rand(total, FLAVS, XGRID)
    subset = err_function.randomize_rep(prior, n)
    return prior, subset


def test_randomize_rep(n=4, total=10):
    """ Ensures that the output of the function
    is indeed a subset of the input """
    prior, subset = get_subset(n, total)
    for replica in subset:
        assert replica in prior
