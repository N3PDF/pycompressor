"""
File that manages the compression algorithm.

For the time being, this compressor select a
random subset from the Prior and compute the ERF.
"""

import numpy as np
from pycompressor.err_function import erfs


def error_function(prior, nb_reduc):
    """
    Take the prior replica and extract nd_reduc
    number of reduced replica. Then compute the
    Error Function.

    Arguments:
    ---------
    - prior   : Prior PDF set of shape (repl,fl,xgrid)
    - nb_reduc: Number of replica in the reduced set
    """
    # Choose a non-redundant subset of replicas
    index = np.random.choice(
            prior.shape[0],
            nb_reduc,
            replace=False
            )
    reduc_rep = prior[index]
    erf_res = erfs(prior, reduc_rep)
    tot_erf = 0
    for key in erf_res:
        print(f'ERF_{key}: {erf_res[key]}')
        tot_erf += erf_res[key]

    return tot_erf / len(erf_res)
