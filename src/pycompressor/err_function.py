"""
Computation of the error function
"""

import numpy as np
from pycompressor.estimators import Estimators


def compute_erfm(set_a, set_b, estm):
    """
    Non-normalized error function. The ERF of
    the moment estimators given by eq.(6) of
    https://arxiv.org/pdf/1504.06469.

    Arguments:
    ---------
    - set_a: Prior set of shape (repl,fl,x)
    - set_a: Reduc/Randm set of shape (repl,fl,x)
    - est  : Name of the estimator
    """
    # Compute Estimators
    est_prior = set_a.compute_for(estm)
    est_reduc = set_b.compute_for(estm)
    flv_size = est_prior.shape[0]
    xgd_size = est_prior.shape[1]
    reslt = 0
    for fl in range(flv_size):
        for xg in range(xgd_size):
            if est_prior[fl][xg] != 0:
                fi = est_reduc[fl][xg]
                gi = est_prior[fl][xg]
                reslt += pow((fi - gi) / gi, 2)
    return reslt


def compute_erfs(set_a, set_b, estm):
    """
    Non-normalized error function for Statistical
    estimators.

    The Kolmogorov-smirnov is given by eq.(13) of
    https://arxiv.org/pdf/1504.06469.

    Arguments:
    ---------
    - set_a: Prior set of shape (repl,fl,x)
    - set_a: Reduc/Randm set of shape (repl,fl,x)
    - est  : Name of the estimator
    """
    # Compute Estimators
    # Return arrays of shape (fl,x,region)
    est_prior = set_a.compute_for(estm)
    est_reduc = set_b.compute_for(estm)
    flv_size = est_prior.shape[0]
    xgd_size = est_prior.shape[1]
    region_size = est_prior.shape[2]
    reslt = 0
    for fl in range(flv_size):
        for xg in range(xgd_size):
            for rg in range(region_size):
                if est_prior[fl][xg][rg] != 0:
                    fi = est_reduc[fl][xg][rg]
                    gi = est_prior[fl][xg][rg]
                    reslt += pow((fi - gi) / gi, 2)
    return reslt


def compute_erfc(set_a, set_b, estm):
    """
    Non-normalized error function for correlation
    estimators.

    The correlation ERF is given by eq.(21) of
    https://arxiv.org/pdf/1504.06469.

    Arguments:
    ---------
    - set_a: Prior set of shape (repl,fl,x)
    - set_a: Reduc/Randm set of shape (repl,fl,x)
    - est  : Name of the estimator
    """
    # Compute Estimators
    # Outputs a Matrix
    est_prior = set_a.compute_for(estm)
    est_reduc = set_b.compute_for(estm)
    # Compute inverse of prior
    prior_inv = np.linalg.inv(est_prior)
    fi = np.dot(est_reduc, prior_inv)
    gi = np.dot(est_prior, prior_inv)
    fi_trace = np.trace(fi)
    gi_trace = np.trace(gi)
    try:
        reslt = pow((fi_trace - gi_trace) / gi_trace, 2)
    except ValueError:
        print("The correlation matrix is incorrect.")
    return reslt


def randomize_rep(replica, number):
    """
    Extract a subset of random replica from
    the prior.

    (Non-redundant choice)
    """
    index = np.random.choice(replica.shape[0], number, replace=False)
    return replica[index]


def compute_cfd68(reslt_trial):
    """
    Compute the 68% confidence interval of
    a randomized trial arrays.

    Arguments:
    ---------
    - reslt_trial: Array of shape of (size_trials)
    """
    size = reslt_trial.shape[0]
    sort = np.sort(reslt_trial)
    cfd_ind = int((size * (1 - 0.68) / 2))
    cfd_val = sort[size - cfd_ind - 1]
    return cfd_val


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

    def __init__(self, prior, reduc, trial=1000):
        self.prepl = prior
        self.trial = trial
        self.prior = Estimators(prior)
        self.reduc = Estimators(reduc)
        # self.rndrp = prior.shape[0] - 1
        self.rndrp = reduc.shape[0]

    def normalize_erfm(self, estm):
        """
        Compute the normalization factor of the class
        Moment Estimator.

        Argument:
        --------
        - estm: Moment estimator name
        """
        reslt = np.zeros(self.trial)
        for t in range(self.trial):
            # Subset of random replica
            # TODO: Check if randomizing the set for each
            # trial is needed
            randm = randomize_rep(self.prepl, self.rndrp)
            est_randm = Estimators(randm)
            # Compute non-normalized erf for each trial
            reslt[t] = compute_erfm(self.prior, est_randm, estm)
        # Compute 68% confidence interval
        normalization = compute_cfd68(reslt)
        return normalization

    def normalize_erfs(self, estm):
        """
        Compute the normalization factor of the class
        Statistical Estimator.

        Argument:
        --------
        - estm: Statistical estimator name
        """
        reslt = np.zeros(self.trial)
        for t in range(self.trial):
            # Subset of random replica
            # TODO: Check if randomizing the set for each
            # trial is needed
            randm = randomize_rep(self.prepl, self.rndrp)
            est_randm = Estimators(randm)
            # Compute non-normalized erf for each trial
            reslt[t] = compute_erfs(self.prior, est_randm, estm)
        # Compute 68% confidence interval
        normalization = compute_cfd68(reslt)
        return normalization

    def normalize_erfc(self, estm):
        """
        Compute the normalization factor of the class
        Correlation Estimator.

        Argument:
        --------
        - estm: Correlation estimator name
        """
        reslt = np.zeros(self.trial)
        for t in range(self.trial):
            # Subset of random replica
            # TODO: Check if randomizing the set for each
            # trial is needed
            randm = randomize_rep(self.prepl, self.rndrp)
            est_randm = Estimators(randm)
            # Compute non-normalized erf for each trial
            reslt[t] = compute_erfc(self.prior, est_randm, estm)
        # Compute 68% confidence interval
        normalization = compute_cfd68(reslt)
        return normalization

    def erfm(self, estm):
        """
        Compute normalized ERF for moment estimators
        """
        nerf = compute_erfm(self.prior, self.reduc, estm)
        norm = self.normalize_erfm(estm)
        return nerf / norm

    def erfs(self, estm):
        """
        Compute normalized ERF for statistical estimators
        """
        nerf = compute_erfs(self.prior, self.reduc, estm)
        norm = self.normalize_erfs(estm)
        return nerf / norm

    def erfc(self, estm):
        """
        Compute normalized ERF for correlation estimators
        """
        nerf = compute_erfm(self.prior, self.reduc, estm)
        norm = self.normalize_erfc(estm)
        return nerf / norm


def erfs(prior, reduc):
    """
    Compute the error function for all the
    estimators.

    Arguments:
    ---------
    - prior_set: Prior/Input replicas
    - reduc_set: Reduced set of replicas
    """
    moment_estimators = [
        "mean",
        "stdev",
        "skewness",
        "kurtosis",
        "moment5th",
        "moment6th",
    ]
    stat_estimators = ["kolmogorov_smirnov"]
    corr_estimators = ["correlation"]
    erf_dic = {}
    # Initialize ERF computation class
    erf = ErfComputation(prior, reduc)

    # Moment Estimators
    for est in moment_estimators:
        erf_estm = erf.erfm(est)
        erf_dic[est] = erf_estm

    # Statistical Estimators
    for est in stat_estimators:
        erf_ests = erf.erfs(est)
        erf_dic[est] = erf_ests

    # # Correlation Estimators
    # for est in corr_estimators:
    #     erf_estc = erf.erfc(est)
    #     erf_dic[est] = erf_estc

    return erf_dic
