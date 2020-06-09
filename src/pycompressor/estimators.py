"""
Collects estimators and corresponding normalizations
"""

import numpy as np
from numba import njit


def replace(input_array):
    """
    Replace values in array

    Parameters
    ----------
        input_array: array
            Array of shape=(flavours, x-grid)

    Returns
    -------
        result: array
            Array of shape=(falvours, x-grid)
    """
    array = input_array
    array[array == 1] = 0
    array[array == -1] = 1
    return array


@njit
def _moment(replicas, mean, stdev, order):
    """ Compute skewness in the standard way following
    exactly eq.(11) of the paper. """
    nrep = replicas.shape[0]
    nflv = replicas.shape[1]
    nxgd = replicas.shape[2]
    result = np.zeros((nflv, nxgd))
    for fl in range(nflv):
        for x in range(nxgd):
            suma = 0
            for rep in range(nrep):
                repl = replicas[rep][fl][x] - mean[fl][x]
                suma += pow(repl, order)
            result[fl][x] = suma / pow(stdev[fl][x], order)
    return result / nrep


# @njit
# def _kolmogorov(replicas, mean, stdev, nbregions=6):
#     """ Compute Kolmogorov-smirnov (KS) estimator as in the C-implementation
#     of the compressor:
#     https://github.com/scarrazza/compressor/blob/master/src/Estimators.cc#L122
#     As opposed to the above implementation, this computes the KS for all
#     replicas, flavours and x-grid. """
#     nrep = replicas.shape[0]
#     nflv = replicas.shape[1]
#     nxgd = replicas.shape[2]
# 
#     def _ks_c_compressor(flavour, x):
#         res = np.zeros(nbregions)
#         for r in range(nrep):
#             val = replicas[r][flavour][x]
#             if val <= mean[flavour][x] - 2 * stdev[flavour][x]:
#                 res[0] += 1
#             elif val <= mean[flavour][x] - 1 * stdev[flavour][x]:
#                 res[1] += 1
#             elif val <= mean[flavour][x]:
#                 res[2] += 1
#             elif val <= mean[flavour][x] + 1 * stdev[flavour][x]:
#                 res[3] += 1
#             elif val <= mean[flavour][x] + 2 * stdev[flavour][x]:
#                 res[4] += 1
#             elif val > mean[flavour][x] + 2 * stdev[flavour][x]:
#                 res[5] += 1
#         return res / nrep
# 
#     result = np.zeros((nflv, nxgd, nbregions))
#     for fl in range(nflv):
#         for x in range(nxgd):
#             result[fl][x] = _ks_c_compressor(fl, x)
#     return result


@njit
def _kolmogorov(replicas, mean, stdev, nb_regions=6):
    """ Compute Kolmogorov-smirnov (KS) estimator as in the C-implementation
    of the compressor:

    https://github.com/scarrazza/compressor/blob/master/src/Estimators.cc#L122

    As opposed to the above implementation, this computes the KS for all
    replicas, flavours and x-grid. """
    nrep = replicas.shape[0]
    nflv = replicas.shape[1]
    nxgd = replicas.shape[2]

    st_ks = np.zeros((nflv, nxgd, nb_regions))
    for fl in range(nflv):
        for x in range(nxgd):
            res = np.zeros(nb_regions)
            for r in range(nrep):
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


@njit
def _correlation(replicas):
    """
    Compute correlation matrix as in eq.(16) of
    https://arxiv.org/pdf/1504.06469.


    NOTE: This algorithm follows exaclty the one
    in the compressor code.
    """
    nrep = replicas.shape[0]
    nflv = replicas.shape[1]
    nxgd = replicas.shape[2]
    # Define nxcorr
    nxcorr = 5
    size = nxcorr * nflv
    # Select x's in the grid
    xs = [int(i / (nxcorr) * nxgd) for i in range(1, nxcorr)]
    xs.append(int(nflv - 1))
    nx = len(xs)
    # Init. Matrix
    cor_mat = np.zeros((size, size))
    for fl1 in range(nflv):
        for x1 in range(nx):
            i = nx * fl1 + x1
            for fl2 in range(nflv):
                for x2 in range(nx):
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


class Estimators:
    """
    Class containing the different types of statistical
    estimators.

    What this class is doing is: take a replica (prior/reduced)
    with a shape (repl,fl,xgrid) and then compute the value of
    the estimators w.r.t to the PDF replicas

    Parameters
    ----------
        replicas: array
            Prior or Reduced PDF replicas of shape=(replicas, flavours, x-grid)
        axs: int
            Axis to which the estimator is computed. By default is set to zero
            to compute along the direction of the pdf replicas

    Returns
    -------
        results: array
    """

    def __init__(self, replicas, axs=0):
        self.axs = axs
        self.replicas = replicas
        self.nrep = replicas.shape[0]
        self.nflv = replicas.shape[1]
        self.nxgd = replicas.shape[2]
        # Compute mean and std estimators first
        # as they are repeatedly called by the
        # other estimators.
        self._mean = np.mean(replicas, axis=0)
        # ddof=1 means that the sum is divided by N-1
        self._stdev = np.std(replicas, axis=0, ddof=1)

    def mean(self):
        return self._mean

    def stdev(self):
        return self._stdev

    def skewness(self):
        return _moment(self.replicas, self._mean, self._stdev, 3)

    def kurtosis(self):
        return _moment(self.replicas, self._mean, self._stdev, 4)

    def moment5th(self):
        """
        Compute the 5th moment
        """
        return _moment(self.replicas, self._mean, self._stdev, 5)

    def moment6th(self):
        """
        Compute the 6th moment
        """
        return _moment(self.replicas, self._mean, self._stdev, 6)

    def kolmogorov_smirnov(self):
        """
        Compute Kolmogorov-smirnov:
        Count the number of replicas (for all fl and x in xgrid) which fall
        in the region given by eq.(14) of https://arxiv.org/abs/1504.06469
        and normalize by the total number of replicas.
        """
        return _kolmogorov(self.replicas, self._mean, self._stdev)

    def correlation(self):
        """
        Compute correlation matrix as in eq.(16) of
        https://arxiv.org/pdf/1504.06469.


        NOTE: This algorithm follows exaclty the one
        in the compressor code.
        """
        return _correlation(self.replicas)

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
            return self.mean()
        elif estm_name == "stdev":
            return self.stdev()
        elif estm_name == "skewness":
            return self.skewness()
        elif estm_name == "kurtosis":
            return self.kurtosis()
        elif estm_name == "moment5th":
            return self.moment5th()
        elif estm_name == "moment6th":
            return self.moment6th()
        elif estm_name == "kolmogorov_smirnov":
            return self.kolmogorov_smirnov()
        elif estm_name == "correlation":
            return self.correlation()
        else:
            raise ValueError(f"{estm_name} is not a valid Estimator.")
