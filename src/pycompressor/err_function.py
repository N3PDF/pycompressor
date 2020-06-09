"""
Computation of the error function
"""

import numpy as np
from numba import njit
from tqdm import trange
from pycompressor.estimators import Estimators


def randomize_rep(replica, number):
    """
    Extract a subset of random replica from
    the prior with a nun-redundant way.

    Parameters
    ----------
        replica: array
            Prior set of replicas shape=(replicas, flavours, x-grid)
        number: int
            Number of subset of replicas

    Returns
    -------
        result: array
            Randomized array of shape=(number, flavours, x-grid)
    """
    index = np.random.choice(replica.shape[0], number, replace=False)
    return replica[index]


def compute_cfd68(reslt_trial):
    """
    Compute the 68% confidence interval of
    a randomized trial arrays.

    Parameters
    ----------
        reslt_trial: array
            Array of shape=(size_trials)

    Returns
    -------
        result: float
            Value of the 68% cfd
    """
    size = reslt_trial.shape[0]
    sort = np.sort(reslt_trial)
    cfd_ind = int((size * (1 - 0.68) / 2))
    cfd_val = sort[size - cfd_ind - 1]
    return cfd_val


@njit
def compute_erfm(prior, nset):
    """
    Non-normalized error function. The ERF of
    the moment estimators given by eq.(6) of
    https://arxiv.org/pdf/1504.06469.

    Parameters
    ----------
        prior: array
            Prior set of replicas of shape=(flavours, x-grid)
        nset : array
            Reduced or Random set of replica shape=(flavours, x-grid)

    Returns
    -------
        result: float
            Value of the error Estimation
    """
    flv_size = prior.shape[0]
    xgd_size = prior.shape[1]
    reslt = 0
    for fl in range(flv_size):
        for xg in range(xgd_size):
            if prior[fl][xg] != 0:
                fi = nset[fl][xg]
                gi = prior[fl][xg]
                reslt += pow((fi - gi) / gi, 2)
    return reslt


@njit
def compute_erfs(prior, nset):
    """
    Non-normalized error function for Statistical
    estimators.

    The Kolmogorov-smirnov is given by eq.(13) of
    https://arxiv.org/pdf/1504.06469.

    Parameters
    ----------
        prior: array
            Prior set of replicas of shape=(flavours, x-grid)
        nset: array
            Array of shape (flavor, x-grid, regions)

    Returns
    -------
        result: float
            Value of the error Estimation
    """
    flv_size = prior.shape[0]
    xgd_size = prior.shape[1]
    region_size = prior.shape[2]
    reslt = 0
    for fl in range(flv_size):
        for xg in range(xgd_size):
            for rg in range(region_size):
                if prior[fl][xg][rg] != 0:
                    fi = nset[fl][xg][rg]
                    gi = prior[fl][xg][rg]
                    reslt += pow((fi - gi) / gi, 2)
    return reslt


@njit
def compute_erfc(prior, nset):
    """
    Non-normalized error function for correlation
    estimators.

    The correlation ERF is given by eq.(21) of
    https://arxiv.org/pdf/1504.06469.

    Parameters
    ----------
        prior: array
            Prior set of replicas of shape=(flavours, x-grid)
        nset: array
            Array of shape (NxCorr*flavors, NxCorr*flavors)

    Returns
    -------
        result: float
            Value of the error Estimation
    """
    # Compute inverse of prior
    prior_inv = np.linalg.inv(prior)
    fi = np.dot(nset, prior_inv)
    gi = np.dot(prior, prior_inv)
    fi_trace = np.trace(fi)
    gi_trace = np.trace(gi)
    # try:
    #     reslt = pow((fi_trace - gi_trace) / gi_trace, 2)
    # except ValueError:
    #     print("The correlation matrix is incorrect.")
    reslt = pow((fi_trace - gi_trace) / gi_trace, 2)
    return reslt


def estimate(prior, est_dic):
    """
    Compute estimators for the PRIOR set.

    Parameters
    ----------
        prior: array
            Prior set of shape=(replicas, flavours, x-grid)
        est_dic: dict
            Contains the list ot all estimators

    Returns
    -------
        result: float
            Array of shape=(flavours, x-grid)
    """
    est_res = Estimators(prior)
    reslt = {}
    for _, est_list in est_dic.items():
        for es in est_list:
            reslt[es] = est_res.compute_for(es)
    return reslt


def normalization(prior, est_prior, rndm_size, est_dic, trials):
    """
    Compute normalization for each Estimator. The normalization
    is computed by calculating the ERF of the given estimator
    for each trials as given generally by eq.(9) of the paper
    (https://arxiv.org/pdf/1504.06469).

    Parameters
    ----------
        prior: array
            Prior set of replica fo shape=(replicas, flavours, x-grid)
        est_prior: dict
            Dictionary containing the values of the estimated results
        rndm_size: int
            Size of random replicas
        est_dic: dict
            Contains the list of estimators
        trials: int
            Number of random trials

    Returns
    -------
        result: float
            Normalization value for each estimator
    """
    print("\n[+] Computing normalization factors...")
    reslt = {}
    for _, est_list in est_dic.items():
        for es in est_list:
            reslt[es] = np.zeros(trials)
    # Loop over the random trials
    with trange(trials) as iter_trial:
        for t in iter_trial:
            iter_trial.set_description("Random Trial")
            randm = randomize_rep(prior, rndm_size)
            est_cl = Estimators(randm)
            # Normalization for Moment Estimators
            for es in est_dic["moment_estimators"]:
                est_randm = est_cl.compute_for(es)
                reslt[es][t] = compute_erfm(est_prior[es], est_randm)
            # Normalization for Statistical Estimators
            for es in est_dic["stat_estimators"]:
                est_randm = est_cl.compute_for(es)
                reslt[es][t] = compute_erfs(est_prior[es], est_randm)
            # Normalization for Statistical Estimators
            for es in est_dic["corr_estimators"]:
                est_randm = est_cl.compute_for(es)
                reslt[es][t] = compute_erfc(est_prior[es], est_randm)
    # Compute 65% confidence interval
    norm = {}
    for est, est_val in reslt.items():
        norm[est] = compute_cfd68(est_val)
        print(" - {:<18} {:^2} {:>}".format(est, ":", norm[est]))
    return norm


class ErfComputation:
    """
    Class that computes the normalized Error Functions.
    The complete ERF expression is generally given by
    eq.(6) in https://arxiv.org/pdf/1504.06469.

    When this class is initialized, the Estimators and
    the normalization factors are computed.

    Parameters
    ----------
        prior: array
            Prior set of replicas of shape=(replicas, flavours, x-grid)
        est_dic: dict
            Contains the list of all the Estimators
        nreduc: int
            Size of reduced replicas
        trials: int
            Number of trials

    Returns
    -------
        result: float
            Normalized ERF
    """

    def __init__(self, prior, est_dic, nreduc, trials=1000):
        self.prior = prior
        self.est_dic = est_dic
        # Compute estimators for PRIOR replicas
        self.pestm = estimate(prior, est_dic)
        # Compute normalizations for each estimator
        self.normz = normalization(prior, self.pestm, nreduc, est_dic, trials)

    def __repr__(self):
        return "Normalizations: {}".format(self.normz)

    def compute_tot_erf(self, reduc):
        """
        Compute the total normalized Error Function which
        is given by the sum of all the normalized estimators.

        Parameters
        ----------
            reduc: array
                Reduced set of replicas of shape=(replica, flavours, x-grid)

        Returns
        -------
            result: float
                Value of the total normalized ERF
        """
        erf = {}
        reduc_cl = Estimators(reduc)
        # Compute non-normalized Moment Estimators
        for es in self.est_dic["moment_estimators"]:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfm(self.pestm[es], est_reduc)
        # Compute non-normalized Statistical Estimators
        for es in self.est_dic["stat_estimators"]:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfs(self.pestm[es], est_reduc)
        # Compute non-normalized Correlation Estimators
        for es in self.est_dic["corr_estimators"]:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfc(self.pestm[es], est_reduc)
        # Normalize ERF
        nerf = 0
        for er in erf.keys():
            nerf += erf[er] / self.normz[er]
        return nerf / len(erf)
