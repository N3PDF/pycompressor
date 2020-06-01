"""
File that manages the compression algorithm.

For the time being, this compressor select a
random subset from the Prior and compute the ERF.
"""

import numpy as np
from pycompressor.err_function import ErfComputation


def error_function(prior, nb_reduc):
    """
    Take the prior replica and extract nd_reduc
    number of reduced replica. Then compute the
    Error Function.

    Arguments:
    ---------
    - prior   : Array of shape=(replicas, flavours, x-grid)
                Prior PDF set of shape (repl,fl,xgrid).
    - nb_reduc: Integer
                Number of replica in the reduced set.

    Outputs: Float
    -------
             Results of ERF for all estimators.
    """
    # Choose a non-redundant subset of replicas
    index = np.random.choice(
            prior.shape[0],
            nb_reduc,
            replace=False
            )
    reduc_rep = prior[index]
    # List of estimators
    est_dic = {
            'moment_estimators': [
                'mean',
                'stdev',
                'skewness',
                'kurtosis'
                ],
            'stat_estimators': [
                'kolmogorov_smirnov'
                ]
            }
    # Compute Normalized Error function
    res = ErfComputation(prior, est_dic, nb_reduc)
    erf_res = res.compute_tot_erf(reduc_rep)
    return erf_res
