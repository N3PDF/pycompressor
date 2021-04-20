# This file contains the definition of the Statistical estimators
# used to evaluate the accuracy of a subset of probability distributions
#
# The definition of the statistical estimators are extracted from the
# paper https://arxiv.org/pdf/1504.06469

import numpy as np
from numba import njit, prange


ALLOWED_ESTIMATORS = [
    "mean",
    "stdev",
    "skewness",
    "kurtosis",
    "moment5th",
    "moment6th",
    "kolmogorov_smirnov",
    "correlation",
]


class Estimators:
    """Class containing the different types of statistical estimators.

    This class takes a set of PDF replicas (prior/compressed/random)
    with a shape (replicas, flavours, xgrid) and then compute the value
    of the estimators w.r.t to the PDF replicas

    Parameters
    ----------
    replicas: array
        Prior or Reduced PDF replicas of shape=(replicas, flavours, x-grid)
    axs: int
        Axis to which the estimator is computed. By default is set to zero
        to compute along the direction of the pdf replicas
    """

    def __init__(self, replicas, axs=0):
        self.axs = axs
        self.replicas = replicas
        # Compute mean and std estimators first
        # as they are repeatedly called by the
        # other estimators.
        self._mean = np.mean(replicas, axis=0)
        # ddof=1 means that the sum is divided by N-1
        self._stdev = np.std(replicas, axis=0, ddof=1)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def moment(replicas, mean, stdev, order):
        """Compute skewness in the standard way following
        exactly eq.(11) of the paper.

        Parameters
        ----------
        replicas: array_like
            Array of PDF replicas (prior/reduced/random)
        mean: array_like
            Array with the mean values of replicas
        stdev: array_like
            Array with the values of standard deviation of replicas
        nb_regions: int, optional
            Number of regions. This is by default set to 6

        Returns
        -------
        array_like
            Array of the value of the n-order moment
        """
        nrep, nflv, nxgd = replicas.shape
        result = np.zeros((nflv, nxgd))
        for fl in prange(nflv):
            for x in prange(nxgd):
                suma = 0
                for rep in prange(nrep):
                    repl = replicas[rep][fl][x] - mean[fl][x]
                    suma += pow(repl, order)
                result[fl][x] = suma / pow(stdev[fl][x], order)
        return result / nrep

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def kolmogorov(replicas, mean, stdev):
        """Compute Kolmogorov-smirnov (KS) estimator as in the C-implementation
        of the compressor:

        https://github.com/scarrazza/compressor/blob/master/src/Estimators.cc#L122

        This function counts the number of replicas (for all fl and x in xgrid)
        which fall in the region given by eq.(14) of https://arxiv.org/abs/1504.06469
        and normalize the result by the total number of replicas.

        As opposed to the above implementation, this computes the KS for all
        replicas, flavours and x-grid.

        Parameters
        ----------
        replicas: array_like
            PDF replicas (prior/reduced/random)
        mean: array_like
            Array with the mean values of replicas
        stdev: array_like
            Array with the values of standard deviation of replicas

        Returns
        -------
            array_like
                Array containing the number of replicas that fall into a region
        """
        nb_regions = 6
        nrep, nflv, nxgd = replicas.shape
        st_ks = np.zeros((nflv, nxgd, nb_regions))
        for fl in prange(nflv):
            for x in prange(nxgd):
                res = np.zeros(nb_regions)
                for r in prange(nrep):
                    val = replicas[r][fl][x]
                    if val <= mean[fl][x] - 2 * stdev[fl][x]:
                        res[0] += 1
                    elif val <= mean[fl][x] - 1 * stdev[fl][x]:
                        res[1] += 1
                    elif val <= mean[fl][x]:
                        res[2] += 1
                    elif val <= mean[fl][x] + 1 * stdev[fl][x]:
                        res[3] += 1
                    elif val <= mean[fl][x] + 2 * stdev[fl][x]:
                        res[4] += 1
                    elif val > mean[fl][x] + 2 * stdev[fl][x]:
                        res[5] += 1
                st_ks[fl][x] = res / nrep
        return st_ks

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def correlation(replicas):
        """Compute the correlation matrix of a given PDF replicas as in eq.(16) of
        https://arxiv.org/pdf/1504.06469.

        Parameters
        ----------
        replicas: array_like
            Array of PDF replicas (prior/reduced/random)

        Returns
        -------
        array_like
            Correlation matrix
        """
        nxcorr = 5
        nrep, nflv, nxgd = replicas.shape
        size = nxcorr * nflv
        # Select x's in the grid
        xs = [int(i / (nxcorr) * nxgd) for i in range(1, nxcorr)]
        xs.append(int(nflv - 1))
        nx = len(xs)
        # Init. Matrix
        cor_mat = np.zeros((size, size))
        for fl1 in prange(nflv):
            for x1 in prange(nx):
                i = nx * fl1 + x1
                for fl2 in prange(nflv):
                    for x2 in prange(nx):
                        j = nx * fl2 + x2
                        sq_i, sq_j = 0, 0
                        i_corr, j_corr, ij_corr = 0, 0, 0
                        for r in range(nrep):
                            x1x, x2x = xs[x1], xs[x2]
                            res1 = replicas[r][fl1][x1x]
                            res2 = replicas[r][fl2][x2x]
                            i_corr += res1
                            j_corr += res2
                            ij_corr += res1 * res2
                            sq_i += res1 * res1
                            sq_j += res2 * res2
                        i_corr /= nrep
                        j_corr /= nrep
                        ij_corr /= nrep
                        # Compute standard deviation
                        fac = nrep - 1
                        std_i = np.sqrt(sq_i / fac - nrep / fac * i_corr * i_corr)
                        std_j = np.sqrt(sq_j / fac - nrep / fac * j_corr * j_corr)
                        # Fill corr. matrix
                        num = ij_corr - (i_corr * j_corr)
                        den = std_i * std_j
                        cor_mat[i][j] = nrep / fac * num / den
        return cor_mat

    def compute_for(self, estm_name):
        """
        Method that maps the called estimators
        to the coorect one.

        Parameters
        ----------
        estm_name: str
            Name of the estimator
        """
        if estm_name == "mean":
            return self._mean
        elif estm_name == "stdev":
            return self._stdev
        elif estm_name == "skewness":
            return self.moment(self.replicas, self._mean, self._stdev, 3)
        elif estm_name == "kurtosis":
            return self.moment(self.replicas, self._mean, self._stdev, 4)
        elif estm_name == "moment5th":
            return self.moment(self.replicas, self._mean, self._stdev, 5)
        elif estm_name == "moment6th":
            return self.moment(self.replicas, self._mean, self._stdev, 6)
        elif estm_name == "kolmogorov_smirnov":
            return self.kolmogorov(self.replicas, self._mean, self._stdev)
        elif estm_name == "correlation":
            return self.correlation(self.replicas)
        else:
            raise ValueError(f"{estm_name} is not a valid Estimator.")
