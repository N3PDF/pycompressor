"""
File that manages the compression algorithm.

For the time being, this compressor select a
random subset from the Prior and compute the ERF.
"""

import numpy as np
from pycompressor.err_function import ErfComputation


class compress:
    """
    Compress the Prior set of replicas into a
    subset of replicas that faithfully contains the
    statistical properties of the prior (in other words
    a subset that gives the best value of the error
    function).

    Parameters
    ----------
    """
    def __init__(self, prior, est_dic, nb_reduc):
        self.prior = prior
        self.est_dic = est_dic
        self.nb_reduc = nb_reduc
        # Init. ErfComputation class. This also computes
        # the one-time computation of the estimators
        # for the prior
        self.err_func = ErfComputation(prior, est_dic, nb_reduc)
        # Init. index for ERF computation
        self.index = np.arange(1, self.nb_reduc + 1, 1)

    def error_function(self, index):
        """
        Sample a subset of replicas as given by the index.
        Then computes the corrresponding ERF value.

        Parameters
        ----------
            index: array
                Array containing the index of the replicas

        Returns
        -------
            result: float
                Value of the ERF
        """
        reduc_rep = self.prior[index]
        # Compute Normalized Error function
        erf_res = self.err_func.compute_tot_erf(reduc_rep)
        return erf_res

    def genetic_algorithm(self, ga_params):
        """
        Look for the combination of replicas that gives the
        best ERF value.

        Parameters
        ----------
            ga_params: dict
                Dictionary containing the inner parameters
                of the GA-ES algorithm

        Returns
        -------
            result: tuple
                'besterf': float
                    Vaue of the best ERF
                'index': array
                    Contains the index of the reduced PDF
        """
        nmut = ga_params['number_mutation']
        # Compute ERF
        berf = self.error_function(self.index)
        # Construc mutation matrix
        mut = np.full((nmut, self.index.shape[0]), self.index)
        # Perform mutation
        for i in range(nmut):
            # Define mutation rate
            r = np.random.uniform(0, 1)
            if r <= .3:
                _nmut = 1
            elif r > .3 and r <= .6:
                _nmut = 2
            elif r > .6 and r <= .7:
                _nmut = 3
            else:
                _nmut = 4
            for _ in range(_nmut):
                p = np.random.randint(1, self.prior.shape[0])
                k = np.random.randint(1, self.nb_reduc)
                mut[i][k] = p
        # Compute ERF for the new sample
        erf = np.zeros(nmut)
        for m in range(nmut):
            erf[m] = self.error_function(mut[m])
        # Perform Selection
        besterf = np.min(erf)                 # Find the lowest ERF
        idx = np.where(erf == besterf)[0][0]  # Find index of the lowest ERF
        # Update index
        if besterf < berf:
            self.index = mut[idx]
        else:
            besterf = berf
        return besterf, self.index

    def cma_algorithm(self):
        pass
