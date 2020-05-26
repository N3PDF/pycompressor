"""
Collects estimators and corresponding normalizations
"""

from math import sqrt
import numpy as np


class Estimators:
    """
    Class containing the different types of statistical
    estimators.

    What this class is doing is: take a replica (prior/reduced)
    with a shape (repl,fl,xgrid) and then compute the value of
    the estimators w.r.t to the PDF replicas.

    Arguments:
    ---------
    - replicas: Prior or Reduced PDF replicas of shape (rep,fl,xgrid)
    - axs     : Axis to which the estimator is computed.
                By default is set to zero to compute
                along the direction of the pdf replicas.
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
        # Compute standard deviation
        rp_stdev = np.std(self.replicas, axis=self.axs)
        return rp_stdev

    def skewness(self):
        # compute skewness value
        rp_mean = self.mean()
        rp_stdev = self.stdev()
        res = 0
        for replica in self.replicas:
            numr = replica - rp_mean
            res += pow(numr, 3)
        return res / (self.nrep * pow(rp_stdev, 3))

    def kurtosis(self):
        # compute kurtosis value
        rp_mean = self.mean()
        rp_stdev = self.stdev()
        res = 0
        for replica in self.replicas:
            numr = replica - rp_mean
            res += pow(numr, 4)
        return res / (self.nrep * pow(rp_stdev, 4))

    def moment5th(self):
        # Compute the 5th moment
        rp_mean = self.mean()
        rp_stdev = self.stdev()
        res = 0
        for replica in self.replicas:
            numr = replica - rp_mean
            res += pow(numr, 5)
        return res / (self.nrep * pow(rp_stdev, 5))

    def moment6th(self):
        # Compute 6th moment
        rp_mean = self.mean()
        rp_stdev = self.stdev()
        res = 0
        for replica in self.replicas:
            numr = replica - rp_mean
            res += pow(numr, 6)
        return res / (self.nrep * pow(rp_stdev, 6))

    def kolmogorov_smirnov(self):
        """
        Compute Kolmogorov-smirnov:
        Count the number of replicas (for all fl and x in xgrid) which fall
        in the region given by eq.(14) of https://arxiv.org/abs/1504.06469
        and normalize by the total number of repplicas.

        Ouput:
        -----
        Array of shape (fl,xgrid,regions)
        """
        region_size = 6
        # Compute mean and std of replica
        rp_mean = self.mean()
        rp_stdev = self.stdev()
        # Initialize array to put the results
        reslt = np.zeros(region_size)
        ks_mat = np.zeros((self.nflv, self.nxgd, region_size))
        for replica in self.replicas:
            for f in range(self.nflv):
                for x in range(self.nxgd):
                    if replica[f][x] <= (rp_mean[f][x] - 2 * rp_stdev[f][x]):
                        reslt[0] += 1
                    elif replica[f][x] <= (rp_mean[f][x] - rp_stdev[f][x]):
                        reslt[1] += 1
                    elif replica[f][x] <= (rp_mean[f][x]):
                        reslt[2] += 1
                    elif replica[f][x] <= (rp_mean[f][x] + rp_stdev[f][x]):
                        reslt[3] += 1
                    elif replica[f][x] <= (rp_mean[f][x] + 2 * rp_stdev[f][x]):
                        reslt[4] += 1
                    elif replica[f][x] > (rp_mean[f][x] + 2 * rp_stdev[f][x]):
                        reslt[5] += 1
                    else:
                        raise Exception("The replica did not fall in the regions.")
                    ks_mat[f][x] = reslt
        return ks_mat / self.nrep

    def correlation(self):
        """
        Compute correlation matrix as in eq.(16) of
        https://arxiv.org/pdf/1504.06469.


        NOTE: This algorithm follows exaclty the one
        in the compressor code.

        Input:
        -----
        Array of shape (repl,fl,xgrid)

        Ouput:
        -----
        Array of shape (NxCorr*flv,NxCorr*flv)
        """
        # Define Nxcorr
        Nxcorr = 5
        size = Nxcorr * self.nflv
        # Select x's in the grid
        xs = [int(i/(Nxcorr)) * self.nxgd for i in range(1, Nxcorr)]
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
                        std_i = sqrt(sq_i / fac - self.nrep / fac * i_corr * i_corr)
                        std_j = sqrt(sq_j / fac - self.nrep / fac * j_corr * j_corr)
                        # Fill corr. matrix
                        num = ij_corr - (i_corr * j_corr)
                        den = std_i * std_j
                        cor_mat[i][j] = self.nrep / fac * num / den
        return cor_mat
