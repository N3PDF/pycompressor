"""
Collects estimators and corresponding normalizations
"""

import numpy as np
from scipy import stats


class Estimators:
    """
    Class that contains the different types of
    statistical estimators.

    The estimators are omputed w.r.t the pdf replicas.

    Arguments:
    ---------
    - prior: Prior/Input PDF replicas
    - reduc: Reduced set of PDF replicas
    - axs  : Axis to which the estimator is computed.
             By default is set to zero to compute
             along the direction of the pdf replicas.
    """

    def __init__(self, prior, reduc, axs=0):
        self.axs = axs
        self.eps = 1e-8
        self.prior = prior
        self.reduc = reduc
        self.nrep = prior.shape[0]

    def mean(self):
        # Compute mean value
        p_mean = np.mean(self.prior, axis=self.axs) + self.eps
        r_mean = np.mean(self.reduc, axis=self.axs) + self.eps
        return p_mean, r_mean

    def stdev(self):
        # Compute standard deviation
        p_std = np.std(self.prior, axis=self.axs) + self.eps
        r_std = np.std(self.reduc, axis=self.axs) + self.eps
        return p_std, r_std

    def skewness(self):
        # compute skewness value
        pass

    # @staticmethod
    # @njit
    # def _mean_arrays(prior, reduc, axs=0, eps=1e-8):
    #     # Compute mean value
    #     p_mean = np.mean(prior, axis=axs) + eps
    #     r_mean = np.mean(reduc, axis=axs) + eps
    #     return p_mean, r_mean

    # @staticmethod
    # @njit
    # def _stdev_arrays(prior, reduc, axs=0, eps=1e-8):
    #     # Compute standard deviation
    #     p_std = np.std(prior, axis=axs) + self.eps
    #     r_std = np.std(reduc, axis=axs) + self.eps
    #     return p_std, r_std

    # @staticmethod
    # @njit
    # def _skewness_arrays(prior, reduc, axs=0, eps=1e-8):
    #     # compute skewness value
    #     pass


class NormalizationK:
    """
    Class that computes the normalization K for a given estimator
    as expressed by eq. (9) of https://arxiv.org/abs/1504.06469.

    Notice that the normalization factor is defined for each estimator.

    Arguments:
    - prior: Prior set of replicas
    - random_param: Radom parameter generation
    """

    def __init__(self, prior, random_param):
        self.prior = prior
        self.random_param = random_param

    def random_replicas(self, number):
        """ Selcet non-redundant replicas from prior """
        index = np.random.choice(self.prior.shape[0], number, replace=False)
        return self.prior[index]

    def cfd68(self, name_est, randm):
        """
        Select replicas that falls into the 68% confidence
        interval
        """
        eps = 1e-8
        estm = Estimators(self.prior, randm, axs=0)
        pr_mean, rd_mean = estm.mean()
        pr_stdv, rd_stdv = estm.stdev()
        # Shift std to avoid std=0
        pr_stdv += eps
        rd_stdv += eps
        # Compute 68% level (this returns a tuple)
        pr_cfd = stats.norm.interval(0.6827, loc=pr_mean, scale=pr_stdv)
        rd_cfd = stats.norm.interval(0.6827, loc=rd_mean, scale=rd_stdv)
        res_pr = np.zeros(self.prior.shape[1])
        res_rd = np.zeros(self.prior.shape[1])
        for z in range(self.prior.shape[1]):
            mask_rd = (rd_cfd[0][z] <= randm[:, z]) * (randm[:, z] <= rd_cfd[1][z])
            mask_pr = (pr_cfd[0][z] <= self.prior[:, z]) * (
                self.prior[:, z] <= pr_cfd[1][z]
            )
            # Apply selection
            new_rd = randm[:, z][mask_rd]
            new_pr = self.prior[:, z][mask_pr]

            cfd_class = Estimators(new_pr, new_rd)
            pr_res, rd_res = getattr(cfd_class, name_est, None)()
            res_pr[z] = pr_res
            res_rd[z] = rd_res
        fin_pr = res_pr + eps
        fin_rd = res_rd + eps

        return fin_pr, fin_rd

    def nk_mean(self):
        """
        Normalization factor for mean estimator
        """
        sum2 = 0
        # Select fixed-sized subset from true
        size_rp = 50
        size_rand = 1000
        for r in range(1, size_rand):
            rand_distr = self.random_replicas(size_rp)
            xpr, xrd = self.cfd68("mean", rand_distr)
            sum1 = ((xrd - xpr) / xpr) ** 2
            sum2 += np.sum(sum1)
        return sum2 / size_rand

    def nk_stdev(self):
        return self.nk_mean()
