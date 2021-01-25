# This file contains the modules that computes the Error Function (ERF)
# by taking the upper 68% from the confidence interval.

import json
import logging
import numpy as np
from numba import njit
from tqdm import trange
from rich.table import Table
from rich.console import Console
from pycompressor.estimators import Estimators

console = Console()
log = logging.getLogger(__name__)


def randomize_rep(replica, number, rndgen):
    """Extract a subset of random replica from the prior in a nun-
    redundant way (no duplicates).

    Parameters
    ----------
        replica: array_like
            Prior set of replicas shape=(replicas, flavours, x-grid)
        number: int
            Number of subset of replicas

    Returns
    -------
        array_like
            Randomized array of shape=(number, flavours, x-grid)
    """
    index = rndgen.choice(replica.shape[0], number, replace=False)
    return replica[index]


def compute_cfd68(reslt_trial):
    """Compute the confidence interval of a randomized trial arrays.

    Parameters
    ----------
        reslt_trial: array_like
            Array of shape=(size_trials)

    Returns
    -------
        array_like
            Value of the cfd
    """
    size = reslt_trial.shape[0]
    sort = np.sort(reslt_trial)
    cfdv = np.zeros(8)
    # cv
    cfdv[0] = np.mean(reslt_trial)
    # 50% cfd
    cfd50 = int((size * (1 - 0.50) / 2))
    cfdv[1] = sort[size - cfd50 - 1]    # upper
    cfdv[2] = sort[cfd50]               # lower
    # 68% cfd
    cfd68 = int((size * (1 - 0.68) / 2))
    cfdv[3] = sort[size - cfd68 - 1]    # upper
    cfdv[4] = sort[cfd68]               # lower
    # 90% cfd
    cfd90 = int((size * (1 - 0.90) / 2))
    cfdv[5] = sort[size - cfd90 - 1]    # upper
    cfdv[6] = sort[cfd90]               # lower
    # md
    if (size % 2) == 0:
        cfdv[7] = (sort[size // 2 - 1] + sort[size // 2]) / 2
    else:
        cfdv[7] = sort[size // 2]
    return cfdv


@njit
def compute_erfm(prior, nset):
    """Non-normalized error function. The ERF of the moment estimators
    given by eq.(6) of https://arxiv.org/pdf/1504.06469.

    Parameters
    ----------
        prior: array_like
            Prior set of replicas of shape=(flavours, x-grid)
        nset : array_like
            Reduced or Random set of replica shape=(flavours, x-grid)

    Returns
    -------
        float
            Value of the error Estimation
    """
    reslt = 0
    flv_size, xgd_size = prior.shape
    for fl in range(flv_size):
        for xg in range(xgd_size):
            if prior[fl][xg] != 0:
                fi = nset[fl][xg]
                gi = prior[fl][xg]
                reslt += pow((fi - gi) / gi, 2)
    return reslt


@njit
def compute_erfs(prior, nset):
    """Non-normalized error function for Statistical estimators.

    The Kolmogorov-smirnov is given by eq.(13) of
    https://arxiv.org/pdf/1504.06469.

    Parameters
    ----------
        prior: array_like
            Prior set of replicas of shape=(flavours, x-grid)
        nset: array_like
            Array of shape (flavor, x-grid, regions)

    Returns
    -------
        float
            Value of the error Estimation
    """
    reslt = 0
    flv_size, xgd_size, region_size = prior.shape
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
    """Non-normalized error function for correlation estimators.

    The correlation ERF is given by eq.(21) of
    https://arxiv.org/pdf/1504.06469.

    Parameters
    ----------
        prior: array_like
            Prior set of replicas of shape=(flavours, x-grid)
        nset: array_like
            Array of shape (NxCorr*flavors, NxCorr*flavors)

    Returns
    -------
        float
            Value of the error Estimation
    """
    # Compute inverse of prior
    prior_inv = np.linalg.inv(prior)
    fi = np.dot(nset, prior_inv)
    gi = np.dot(prior, prior_inv)
    fi_trace = np.trace(fi)
    gi_trace = np.trace(gi)
    reslt = pow((fi_trace - gi_trace) / gi_trace, 2)
    return reslt


def estimate(prior, est_dic):
    """Compute estimators for the PRIOR set.

    Parameters
    ----------
        prior: array_like
            Prior set of shape=(replicas, flavours, x-grid)
        est_dic: dict
            Contains the list ot all estimators

    Returns
    -------
        float
            Array of shape=(flavours, x-grid)
    """
    est_res = Estimators(prior)
    reslt = {}
    for _, est_list in est_dic.items():
        for es in est_list:
            reslt[es] = est_res.compute_for(es)
    return reslt


def normalization(prior, est_prior, rndm_size, est_dic, trials, folder, rndgen):
    """Compute normalization for each Estimator. The normalization is computed
    by calculating the ERF of the given estimator for each trials as given
    generally by eq.(9) of the paper (https://arxiv.org/pdf/1504.06469).

    Parameters
    ----------
        prior: array_like
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
        float
            Normalization value for each estimator
    """
    reslt = {}
    for _, est_list in est_dic.items():
        for es in est_list:
            reslt[es] = np.zeros(trials)
    # Loop over the random trials
    console.print("\n• Evaluate estimators for random sampling:", style="bold blue")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Estimators", justify="left", width=24)
    table.add_column("Values from random sampling", justify="left", width=50)
    with trange(trials) as iter_trial:
        for t in iter_trial:
            iter_trial.set_description("Random Trial")
            randm = randomize_rep(prior, rndm_size, rndgen)
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
    norm, rnderfs_dic = {}, {}
    erfile = open(f"{folder}/erf_randomized.dat", "a+")
    erfile.write(f"{rndm_size}:")
    for est, est_val in reslt.items():
        ucfd68 = compute_cfd68(est_val)
        norm[est] = ucfd68[3]
        rnderfs_dic[est] = {
                "rcv": ucfd68[0],  # CV
                "rmd": ucfd68[7],  # MD
                "l50": ucfd68[2],  # lower 50
                "u50": ucfd68[1],  # upper 50
                "l68": ucfd68[4],  # lower 68
                "u68": ucfd68[3],  # upper 68
                "l90": ucfd68[6],  # lower 90
                "u90": ucfd68[5],  # upper 90
        }
        table.add_row(f"{est}", f"{norm[est]:.4e}")
    console.print(table)
    erfile.write(json.dumps(rnderfs_dic))
    erfile.write("\n")
    erfile.close()
    return norm


class ErfComputation:
    """Class that computes the normalized Error Functions. The complete ERF
    expression is generally given by eq.(6) of https://arxiv.org/pdf/1504.06469.

    When this class is initialized, the Estimators and the normalization factors
    are computed.

    Parameters
    ----------
        prior: array_like
            Prior set of replicas of shape=(replicas, flavours, x-grid)
        est_dic: dict
            Contains the list of all the Estimators
        nreduc: int
            Size of reduced replicas
        trials: int
            Number of trials
    """

    def __init__(self, prior, est_dic, nreduc, folder, rndgen, trials=1000, norm=True):
        self.prior = prior
        self.est_dic = est_dic
        # Compute estimators for PRIOR replicas
        self.pestm = estimate(prior, est_dic)
        # Compute normalizations for each estimator
        if norm:
            self.normz = normalization(
                    prior,
                    self.pestm,
                    nreduc,
                    est_dic,
                    trials,
                    folder,
                    rndgen
            )

    def __repr__(self):
        return "Normalizations: {}".format(self.normz)

    def compute_tot_erf(self, reduc):
        """Compute the total normalized Error Function which is given by the sum of
        all the normalized estimators.

        Parameters
        ----------
            reduc: array_like
                Reduced set of replicas of shape=(replica, flavours, x-grid)

        Returns
        -------
            float
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

    def compute_all_erf(self, reduc):
        """Compute the total normalized Error Function which is given by the sum of
        all the normalized estimators.

        Parameters
        ----------
            reduc: array_like
                Reduced set of replicas of shape=(replica, flavours, x-grid)

        Returns
        -------
            float
                Value of the total normalized ERF
        """
        erf = {}
        reduc_cl = Estimators(reduc)
        # Compute non-normalized Moment Estimators
        for es in self.est_dic["moment_estimators"]:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfm(self.pestm[es], est_reduc)
            # erf[es] /= self.normz[es]
        # Compute non-normalized Statistical Estimators
        for es in self.est_dic["stat_estimators"]:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfs(self.pestm[es], est_reduc)
            # erf[es] /= self.normz[es]
        # Compute non-normalized Correlation Estimators
        for es in self.est_dic["corr_estimators"]:
            est_reduc = reduc_cl.compute_for(es)
            erf[es] = compute_erfc(self.pestm[es], est_reduc)
            # erf[es] /= self.normz[es]
        return erf
