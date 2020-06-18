# This file contains the definition of the Statistical estimators
# used to evaluate the accuracy of a subset of probability distributions
#
# The definition of the statistical estimators are extracted from the
# paper https://arxiv.org/pdf/1504.06469

import numpy as np
from numba import njit


def replace(input_array):
    """Replace values in array in the following way: change all the
    (1) to (0) and all (-1) to (1).

    Parameters
    ----------
        input_array: array
            Array of shape=(flavours, x-grid)

    Returns
    -------
        array
            Array of shape=(falvours, x-grid)
    """
    array = input_array
    array[array == 1] = 0
    array[array == -1] = 1
    return array


@njit
def _moment(replicas, mean, stdev, order):
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


@njit
def _kolmogorov(replicas, mean, stdev, nb_regions=6):
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
        nb_regions: int, optional
            Number of regions. This is by default set to 6

    Returns
    -------
        array_like
            Array containing the number of replicas that fall into a region
    """
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
    """ Compute the correlation matrix of a given PDF replicas as in eq.(16) of
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
        """Compute the mean of a PDF set along the replicas.

        Returns
        -------
            array_like
                Array of the mean values of replicas
        """
        return self._mean

    def stdev(self):
        """Compute the standard deviation of a PDF set along the replicas.

        Returns
        -------
            array_like
                Array of the standard deviation values of replicas
        """
        return self._stdev

    def skewness(self):
        """Compute the skewness value of a PDF set along the replicas.

        Returns
        -------
            array_like
                Array of the skewness values of replicas
        """
        return _moment(self.replicas, self._mean, self._stdev, 3)

    def kurtosis(self):
        return _moment(self.replicas, self._mean, self._stdev, 4)

    def moment5th(self):
        """Compute the 5th moment of a PDF set along the replicas.

        Returns
        -------
            array_like
                Array of the values of the 5th moment of replicas
        """
        return _moment(self.replicas, self._mean, self._stdev, 5)

    def moment6th(self):
        """Compute the 6th moment of a PDF set along the replicas.

        Returns
        -------
            result: array_like
                Array of the values of the 6th moment of replicas
        """
        return _moment(self.replicas, self._mean, self._stdev, 6)

    def kolmogorov_smirnov(self):
        """Compute the value of the Kolmogorov-smirnov.

        Returns
        -------
            array
                Array containing the number of replicas that fall into a
                region.
        """
        return _kolmogorov(self.replicas, self._mean, self._stdev)

    def correlation(self):
        """Compute the correlation matrix of a given PDF replicas.

        Returns
        -------
            array_like
                Correlation matrix
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
