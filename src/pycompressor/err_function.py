"""
Computation of the error function
"""

import numpy as np
from pycompressor.estimators import Estimators


def randomize_rep(replica, number):
    """
    Extract a subset of random replica from
    the prior with a nun-redundent way.

    Arguments:
    ---------
    - replica: Array of shape=(replicas, flavours, x-grid)
               Prior set of replicas.
    - number : Integer
               Number of subset of replicas.

    Outputs: Array of shape=(number, flavours, x-grid)
    -------
    """
    index = np.random.choice(replica.shape[0], number, replace=False)
    return replica[index]


def compute_cfd68(reslt_trial):
    """
    Compute the 68% confidence interval of
    a randomized trial arrays.

    Arguments:
    ---------
    - reslt_trial: Array of shape=(size_trials)

    Outputs: Float
    -------
    """
    size = reslt_trial.shape[0]
    sort = np.sort(reslt_trial)
    cfd_ind = int((size * (1 - 0.68) / 2))
    cfd_val = sort[size - cfd_ind - 1]
    return cfd_val


def compute_erfm(prior, nset):
    """
    Non-normalized error function. The ERF of
    the moment estimators given by eq.(6) of
    https://arxiv.org/pdf/1504.06469.

    Arguments:
    ---------
    - prior: Array of shape=(flavours, x-grid)
             Prior set of replicas.
    - nset : Array of shape=(flavours, x-gird)
             Reduced or Random set of replica.

    Outputs: Float
    -------
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


def compute_erfs(prior, nset):
    """
    Non-normalized error function for Statistical
    estimators.

    The Kolmogorov-smirnov is given by eq.(13) of
    https://arxiv.org/pdf/1504.06469.

    Arguments:
    ---------
    - prior: Array of shape=(flavours, x-grid)
             Prior set of replicas.
    - nset : Array of shape=(flavours, x-gird)
             Array of shape (flavor, x-grid, regions)

    Outputs: Float
    -------
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


def compute_erfc(prior, nset):
    """
    Non-normalized error function for correlation
    estimators.

    The correlation ERF is given by eq.(21) of
    https://arxiv.org/pdf/1504.06469.

    Arguments:
    ---------
    - prior: Array of shape=(flavours, x-grid)
             Prior set of replicas.
    - nset : Array of shape=(flavours, x-gird)
             Array of shape (NxCorr*flavors, NxCorr*flavors)

    Outputs: Float
    -------
             Value of the error Estimation
    """
    # Compute inverse of prior
    prior_inv = np.linalg.inv(prior)
    fi = np.dot(nset, prior_inv)
    gi = np.dot(prior, prior_inv)
    fi_trace = np.trace(fi)
    gi_trace = np.trace(gi)
    try:
        reslt = pow((fi_trace - gi_trace) / gi_trace, 2)
    except ValueError:
        print("The correlation matrix is incorrect.")
    return reslt


def estimate(prior, est_dic):
    """
    Compute estimators for the PRIOR set.

    Arguments:
    ---------
    - prior  : Array of shape=(replicas, flavours, x-grid)
    - est_dic: Dictionary
               Contains the list ot all estimators

    Outputs: Array of shape=(flavours, x-grid)
    -------
    """
    est_res = Estimators(prior)
    reslt = {}
    for es_type in est_dic.keys():
        for es in est_dic[es_type]:
            reslt[es] = est_res.compute_for(es)
    return reslt


def normalization(prior, est_prior, rndm_size, est_dic, trials):
    """
    Compute normalization for each Estimator. The normalization
    is computed by calculating the ERF of the given estimator
    for each trials as given generally by eq.(9) of the paper
    (https://arxiv.org/pdf/1504.06469).

    Arguments:
    ---------
    - prior    : Array of shape (replicas, flavours, x-grid)
                 Prior set of replica
    - est_prior: Dictionary
                 Contains the values of the estimated results
    - rndm_size: Integer
                 Size of random replicas
    - est_dic  : Dictionary
                 Contains the list of estimators
    - trials   : Integer
                 Number of random trials

    Output: Dictionary of floats
    ------
            Normalization value for each estimator.
    """
    reslt = {}
    for es_class in est_dic.keys():
        for es in est_dic[es_class]:
            reslt[es] = np.zeros(trials)
    # Loop over the random trials
    for t in range(trials):
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
        # # Normalization for Statistical Estimators
        # for es in est_dic['corr_estimators']:
        #     est_randm = est_cl.compute_for(es)
        #     reslt[es][t] = compute_erfc(est_prior[es], est_randm)
    # Compute 65% confidence interval
    norm = {}
    for est in reslt.keys():
        norm[est] = compute_cfd68(reslt[est])
    return norm


class ErfComputation:
    """
    Class that computes the normalized Error Functions.
    The complete ERF expression is generally given by
    eq.(6) in https://arxiv.org/pdf/1504.06469.

    When this class is initialized, the Estimators and
    the normalization factors are computed.

    Arguments:
    ---------
    - prior  : Array os shape=(replicas, flavours, x-grid)
               Prior set of replicas
    - est_dic: Dictionary
               Contains the list of all the Estimators
    - nreduc : Integer
               Size of reduced replicas
    - trials : Integer
               Number of trials

    Outputs:
    -------
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

        Arguments:
        ---------
        - reduc: Array of shape=(replica, flavours, x-grid)
                 Reduced set of replicas.

        Outputs: Float
        -------
                Value of the total normalized ERF
        """
        erf = {}
        reduc_cl = Estimators(reduc)
        # Compute non-normalized Moment Estimators
        for es in self.est_dic['moment_estimators']:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfm(self.pestm[es], est_reduc)
        # Compute non-normalized Statistical Estimators
        for es in self.est_dic['stat_estimators']:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfs(self.pestm[es], est_reduc)
        # # Compute non-normalized Correlation Estimators
        # for es in self.est_dic['corr_estimators']:
        #     est_reduc = reduc_cl.compute_for(es)
        #     erf[es] = compute_erfc(self.pestm[es], est_reduc)
        # Normalize ERF
        nerf = 0
        for er in erf.keys():
            nerf += erf[er] / self.normz[er]
        return nerf / len(erf)
