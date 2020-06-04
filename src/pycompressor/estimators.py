"""
Collects estimators and corresponding normalizations
"""

import numpy as np
from scipy.stats import skew
from scipy.stats import moment
from scipy.stats import kurtosis


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

    def mean(self):
        # Compute mean value
        rp_mean = np.mean(self.replicas, axis=self.axs)
        return rp_mean

    def stdev(self):
        # Compute standard deviation for SAMPLES
        # ddof=1 means that the sum is divided by N-1
        rp_stdev = np.std(self.replicas, axis=self.axs, ddof=1)
        return rp_stdev

    def skewness(self):
        rp_skew = skew(self.replicas, axis=self.axs)
        return rp_skew

    def kurtosis(self):
        rp_kurt = kurtosis(self.replicas, axis=self.axs)
        return rp_kurt

    def moment5th(self):
        """
        Compute the 5th moment

        The computation of the 5th moment in scipy is
        defined here https://docs.scipy.org/doc/scipy/reference/
        generated/scipy.stats.moment.html which is different from
        the one defined here https://github.com/scarrazza/compressor/
        blob/master/src/Estimators.cc#L102 in the sense that it is
        divided by the standard deviation.
        """
        rp_stdev = self.stdev()
        rp_momnt = moment(self.replicas, moment=5, axis=self.axs)
        return rp_momnt / pow(rp_stdev, 5)

    def moment6th(self):
        """
        Compute the 6th moment
        """
        rp_stdev = self.stdev()
        rp_momnt = moment(self.replicas, moment=6, axis=self.axs)
        return rp_momnt / pow(rp_stdev, 6)

    def kolmogorov_smirnov(self):
        """
        Compute Kolmogorov-smirnov:
        Count the number of replicas (for all fl and x in xgrid) which fall
        in the region given by eq.(14) of https://arxiv.org/abs/1504.06469
        and normalize by the total number of replicas.
        """
        region_size = 6
        rp_mean = self.mean()
        rp_stdev = self.stdev()
        # Init. results of comp. with null arrays
        results = {}
        for rs in range(region_size):
            results[f"result_{rs}"] = np.zeros((self.nflv, self.nxgd))
        # Define regions
        regions = {
            "region_0": rp_mean - 2 * rp_stdev,
            "region_1": rp_mean - 1 * rp_stdev,
            "region_2": rp_mean,
            "region_3": rp_mean + 1 * rp_stdev,
            "region_4": rp_mean + 2 * rp_stdev,
            "region_5": rp_mean + 2 * rp_stdev,
        }
        # Loop over replicas
        for replica in self.replicas:
            # 1st region
            bool0 = replica <= regions["region_0"]
            resb0 = bool0.astype(np.int)
            results["result_0"] += resb0
            # 2nd region
            bool1 = replica <= regions["region_1"]
            resb1 = bool1.astype(np.int)
            relb1 = resb0 - resb1
            resg1 = replace(relb1)
            results["result_1"] += resg1
            # 3rd region
            bool2 = replica <= regions["region_2"]
            resb2 = bool2.astype(np.int)
            relb2 = resb0 + resg1 - resb2
            resg2 = replace(relb2)
            results["result_2"] += resg2
            # 4th region
            bool3 = replica <= regions["region_3"]
            resb3 = bool3.astype(np.int)
            relb3 = resb0 + resg1 + resg2 - resb3
            resg3 = replace(relb3)
            results["result_3"] += resg3
            # 5th region
            bool4 = replica <= regions["region_4"]
            resb4 = bool4.astype(np.int)
            relb4 = resb0 + resg1 + resg2 + resg3 - resb4
            resg4 = replace(relb4)
            results["result_4"] += resg4
            # 6th region
            bool5 = replica > regions["region_5"]
            resb5 = bool5.astype(np.int)
            relb5 = resb0 + resg1 + resg2 + resg3 + resg4 - resb5
            resg5 = replace(relb5)
            results["result_5"] += resg5
        rslt = [results[f"result_{i}"] for i in range(region_size)]
        tupl = tuple(rslt)
        fin_res = np.stack(tupl, axis=-1)

        return fin_res / self.nrep

    def correlation(self):
        """
        Compute correlation matrix as in eq.(16) of
        https://arxiv.org/pdf/1504.06469.


        NOTE: This algorithm follows exaclty the one
        in the compressor code.
        """
        # Define nxcorr
        nxcorr = 5
        size = nxcorr * self.nflv
        # Select x's in the grid
        xs = [int(i / (nxcorr)) * self.nxgd for i in range(1, nxcorr)]
        xs.append(int(self.nflv - 1))
        nx = len(xs)
        # Init. Matrix
        cor_mat = np.zeros((size, size))
        for fl1 in range(self.nflv):
            for x1 in range(nx):
                i = nx * fl1 + x1
                for fl2 in range(self.nflv):
                    for x2 in range(nx):
                        j = nx * fl2 + x2
                        sq_i, sq_j = 0, 0
                        i_corr, j_corr, ij_corr = 0, 0, 0
                        for r in range(self.nrep):
                            x1x, x2x = xs[x1], xs[x2]
                            res1 = self.replicas[r][fl1][x1x]
                            res2 = self.replicas[r][fl2][x2x]
                            i_corr += res1
                            j_corr += res2
                            ij_corr += res1 * res2
                            sq_i += res1 * res1
                            sq_j += res2 * res2
                        i_corr /= self.nrep
                        j_corr /= self.nrep
                        ij_corr /= self.nrep
                        # Compute standard deviation
                        fac = self.nrep - 1
                        std_i = np.sqrt(sq_i / fac - self.nrep / fac * i_corr * i_corr)
                        std_j = np.sqrt(sq_j / fac - self.nrep / fac * j_corr * j_corr)
                        # Fill corr. matrix
                        num = ij_corr - (i_corr * j_corr)
                        den = std_i * std_j
                        cor_mat[i][j] = self.nrep / fac * num / den
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
        if estm_name == 'mean':
            return self.mean()
        elif estm_name == 'stdev':
            return self.stdev()
        elif estm_name == 'skewness':
            return self.skewness()
        elif estm_name == 'kurtosis':
            return self.kurtosis()
        elif estm_name == 'moment5th':
            return self.moment5th()
        elif estm_name == 'moment6th':
            return self.moment6th()
        elif estm_name == 'kolmogorov_smirnov':
            return self.kolmogorov_smirnov()
        elif estm_name == 'correlation':
            return self.correlation()
        else:
            raise ValueError(f'{estm_name} is not a valid Estimator.')
