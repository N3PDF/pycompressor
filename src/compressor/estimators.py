import numpy as np
from scipy import stats


class estimators(object):
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
        self.nrep  = prior.shape[0]

    def mean(self):
        # Compute mean value
        Pmean = np.mean(self.prior, axis=self.axs) + self.eps
        Rmean = np.mean(self.reduc, axis=self.axs) + self.eps
        return Pmean, Rmean

    def stdev(self):
        # Compute standard deviation
        Pstd = np.std(self.prior, axis=self.axs) + self.eps
        Rstd = np.std(self.reduc, axis=self.axs) + self.eps
        return Pstd, Rstd

    def skewness(self):
        # compute skewness value
        pass



class normalizationK(object):
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
        # Non-redundant choice
        index = np.random.choice(
            self.prior.shape[0],
            number,
            replace=False
        )
        return self.prior[index]

    def cfd68(self, name_est, randm):
        eps = 1e-8
        estm = estimators(self.prior, randm, axs=0)
        pr_mean, rd_mean = estm.mean()
        pr_stdv, rd_stdv = estm.stdev()
        # Shift std to avoid std=0
        pr_stdv += eps
        rd_stdv += eps
        # Compute 68% level (this returns a tuple)
        pr_cfd = stats.norm.interval(
            0.6827,
            loc=pr_mean,
            scale=pr_stdv
        )
        rd_cfd = stats.norm.interval(
            0.6827,
            loc=rd_mean,
            scale=rd_stdv
        )
        res_pr = np.zeros(self.prior.shape[1])
        res_rd = np.zeros(self.prior.shape[1])
        for z in range(self.prior.shape[1]):
            mask_rd = (rd_cfd[0][z]<=randm[:,z]) * (randm[:,z]<=rd_cfd[1][z])
            mask_pr = (pr_cfd[0][z]<=self.prior[:,z]) * (self.prior[:,z]<=pr_cfd[1][z])
            # Apply selection
            new_rd = randm[:,z][mask_rd]
            new_pr = self.prior[:,z][mask_pr]

            cfd_class = estimators(
                    new_pr,
                    new_rd
            )
            pr_res, rd_res = getattr(cfd_class, name_est, None)()
            res_pr[z] = pr_res
            res_rd[z] = rd_res
        fin_pr = res_pr + eps
        fin_rd = res_rd + eps

        return fin_pr, fin_rd

    def Nk_mean(self):
        """
        Normalization factor for mean estimator
        """
        sum2  = 0
        # Select fixed-sized subset from true
        Nsize = 50
        Nrand = 1000
        for r in range(1, Nrand):
            rand_distr = self.random_replicas(Nsize)
            xpr, xrd = self.cfd68('mean', rand_distr)
            sum1 = ((xrd - xpr) / xpr)**2
            sum2 += np.sum(sum1)
        return sum2/Nrand

    def Nk_stdev(self):
        return self.Nk_mean()
