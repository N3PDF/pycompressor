"""
Computation of the error function
"""

import numpy as np
from compressor.estimators import Estimators


class ErfComputation:
    """
    Class that computes non-normalized error
    function.

    Arguments:
    ---------
    - prior_set: Prior "Estimated" PDF set of shape
                 (replicas,fl,x-gird)
    - reduc_set: Reduced "Estimated" PDF set of shape
                 (replicas,fl,x-grid)
    """

    def __init__(self, prior, reduc):
        self.prior = Estimators(prior)
        self.reduc = Estimators(reduc)
        self.nflv = prior.shape[1]
        self.xgrd = prior.shape[2]

    def compute_erfm(self, estm):
        """
        Non-normalized error function. The ERF of
        the moment estimators is given by eq.(6) of
        https://arxiv.org/pdf/1504.06469.

        Arguments:
        ---------
        - est: Name of the estimator
        """
        # Compute Estimators
        est_prior = getattr(self.prior, estm, None)()
        est_reduc = getattr(self.reduc, estm, None)()
        reslt = 0
        for fl in range(self.nflv):
            for xg in range(self.xgrd):
                if est_prior != 0:
                    fi = est_reduc[fl][xg]
                    gi = est_prior[fl][xg]
                    reslt += pow((fi - gi) / gi, 2)
        return reslt

    def compute_erfs(self, estm):
        """
        Non-normalized error function for Statistical
        estimators.

        The Kolmogorov-smirnov is given by eq.(13) of
        https://arxiv.org/pdf/1504.06469.

        Arguments:
        ---------
        """
        # Compute Estimators
        # Return arrays of shape (fl,x,region)
        est_prior = getattr(self.prior, estm, None)()
        est_reduc = getattr(self.reduc, estm, None)()
        region_size = est_prior.shape[2]
        reslt = 0
        for fl in range(self.nflv):
            for xg in range(self.xgrd):
                for rg in range(region_size):
                    if est_prior[fl][xg][rg] != 0:
                        fi = est_prior[fl][xg][rg]
                        gi = est_reduc[fl][xg][rg]
                        reslt += pow((fi - gi) / gi, 2)
        return reslt

    def compute_erfc(self, estm):
        """
        Non-normalized error function for correlation
        estimators.

        The correlation ERF is given by eq.(21) of
        https://arxiv.org/pdf/1504.06469.

        Arguments:
        ---------
        - est: Name of the estimator
        """
        # Compute Estimators
        # Outputs a Matrix
        est_prior = getattr(self.prior, estm, None)()
        est_reduc = getattr(self.reduc, estm, None)()
        # Compute inverse of prior
        prior_inv = np.linalg.inv(est_prior)
        fi = np.dot(est_prior, prior_inv)
        gi = np.dot(est_reduc, prior_inv)
        fi_trace = np.trace(fi)
        gi_trace = np.trace(gi)
        try:
            reslt = pow((gi_trace - fi_trace) / fi_trace, 2)
        except ValueError:
            print("The correlation matrix is incorrect.")
        return reslt


def erfs(prior, reduc):
    """
    Compute the error function for all the
    estimators.

    Arguments:
    ---------
    - prior_set: Prior/Input replicas
    - reduc_set: Reduced set of replicas
    """
    MomentEstimators = [
        "mean",
        "stdev",
        "skewness",
        "kurtosis",
        "moment5th",
        "moment6th",
    ]
    StatEstimator = ["kolmogorov_smirnov"]
    CorrEstimator = ["correlation"]
    erf_dic = {}
    # Initialize ERF computation class
    erf = ErfComputation(prior, reduc)
    for est in MomentEstimators:
        erf_est = erf.compute_erfm(est)
        # erf_est /= normalizationM(prior, est)
        erf_dic[est] = erf_est

    for est in StatEstimator:
        pass

    for est in CorrEstimator:
        pass
