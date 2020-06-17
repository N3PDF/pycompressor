"""
File that manages the compression algorithm.

For the time being, this compressor select a
random subset from the Prior and compute the ERF.
"""

import cma
import numpy as np
from functools import partial
from multiprocessing import Pool
from pycompressor.err_function import ErfComputation


def cma_erf(index, err_comp, prior):
    """
    Top level function.
    Define the ERF function that is going to be minimized.

    Parameters
    ----------
        index: array
            Array containing the index of the replicas

    Returns
    -------
        result: float
            Value of the ERF
    """
    # Convert float array into int
    index_int = index.astype(int)
    index_modulo = index_int % prior.shape[0]
    # Check Duplicates and if so returns NaN
    duplicates, counts = np.unique(index_modulo, return_counts=True)
    if duplicates[counts > 1].shape[0] != 0:
        return np.NaN
    reduc_rep = prior[index_modulo]
    # Compute Normalized Error function
    erf_res = err_comp.compute_tot_erf(reduc_rep)
    return erf_res


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

    def final_erfs(self, index):
        """
        Compute the final ERF after minimization.

        Parameters
        ----------
            index: array
                Array containing the index of the selected
                replicas

        Returns
        -------
            result: dict
                Dictionary containing the list of estimators
                and their respective values
        """
        selected_replicas = self.prior[index]
        erfs = self.err_func.compute_all_erf(selected_replicas)
        return erfs

    def genetic_algorithm(self, nb_mut=5):
        """
        Look for the combination of replicas that gives the
        best ERF value.

        Parameters
        ----------
            nb_mut: int
                Number of mutation

        Returns
        -------
            result: tuple
                'besterf': float
                    Vaue of the best ERF
                'index': array
                    Contains the index of the reduced PDF
        """
        nmut = nb_mut
        # Compute ERF
        berf = self.error_function(self.index)
        # Construc mutation matrix
        mut = np.full((nmut, self.index.shape[0]), self.index)
        # Perform mutation
        for i in range(nmut):
            # Define mutation rate
            r = np.random.uniform(0, 1)
            if r <= 0.3:
                _nmut = 1
            elif r > 0.3 and r <= 0.6:
                _nmut = 2
            elif r > 0.6 and r <= 0.7:
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
        besterf = np.min(erf)  # Find the lowest ERF
        idx = np.where(erf == besterf)[0][0]  # Find index of the lowest ERF
        # Update index
        if besterf < berf:
            self.index = mut[idx]
        else:
            besterf = berf
        return besterf, self.index

    def cma_algorithm(
        self,
        std_dev=0.3,
        seed=0,
        verbosity=0,
        min_itereval=1000,
        max_itereval=15000
    ):
        """
        Define the ERF function that is going to be minimized.

        Parameters
        ----------
            index: array
                Array containing the index of the replicas

        Returns
        -------
            result: float
                Value of the ERF
        """
        init_index = np.random.choice(
                self.prior.shape[0],
                self.nb_reduc,
                replace=False
        )

        def minimize_erf(index):
            """
            Define the ERF function that is going to be minimized.

            Parameters
            ----------
                index: array
                    Array containing the index of the replicas

            Returns
            -------
                result: float
                    Value of the ERF
            """
            # Convert float array into int
            index_int = index.astype(int)
            index_modulo = index_int % self.prior.shape[0]
            # Check Duplicates and if so returns NaN
            duplicates, counts = np.unique(index_modulo, return_counts=True)
            if duplicates[counts > 1].shape[0] != 0:
                return np.NaN
            reduc_rep = self.prior[index_modulo]
            # Compute Normalized Error function
            erf_res = self.err_func.compute_tot_erf(reduc_rep)
            return erf_res

        # Init CMA class
        options = {"maxiter": max_itereval, "seed": seed, "verb_log": 0}
        cma_es = cma.CMAEvolutionStrategy(init_index, std_dev, options)
        count_it = 0
        while not cma_es.stop() and cma_es.best.f > 2e-2:
            count_it += 1
            pop_solutions, erf_values = [], []
            while len(pop_solutions) < cma_es.popsize:
                current_erf = ind = np.NaN
                while np.isnan(current_erf):
                    ind = cma_es.ask(1)[0]
                    current_erf = minimize_erf(ind)
                pop_solutions.append(ind)
                erf_values.append(current_erf)
            cma_es.tell(pop_solutions, erf_values)
            cma_es.disp()

        # Compute final ERF from selected indices
        cma_res = cma_es.result[0]
        selected_index = cma_res.astype(int)
        selected_modulo = selected_index % self.prior.shape[0]
        erf_cma = self.error_function(selected_modulo)
        return erf_cma, selected_modulo

    def cma_algorithm_multiprocessed(
        self,
        std_dev=0.3,
        seed=0,
        verbosity=0,
        min_itereval=1000,
        max_itereval=15000
    ):
        """
        Define the ERF function that is going to be minimized.

        Parameters
        ----------
            index: array
                Array containing the index of the replicas

        Returns
        -------
            result: float
                Value of the ERF
        """
        init_index = np.random.choice(
                self.prior.shape[0],
                self.nb_reduc,
                replace=False
        )

        # Init CMA class
        options = {"maxiter": max_itereval, "seed": seed, "verb_log": 0}
        cma_es = cma.CMAEvolutionStrategy(init_index, std_dev, options)
        count_it = 0
        with Pool(processes=12) as pool:
            while not cma_es.stop() and cma_es.best.f > 2e-2:
                count_it += 1
                pop_solutions, erf_values = [], []
                while len(pop_solutions) < cma_es.popsize:
                    ind = cma_es.ask(cma_es.popsize - len(pop_solutions))
                    current_erf = pool.map_async(
                        partial(
                            cma_erf, err_comp=self.err_func, prior=self.prior
                        ), ind
                    ).get()
                    for erfval, solution in zip(current_erf, ind):
                        if not np.isnan(erfval):
                            erf_values.append(erfval)
                            pop_solutions.append(solution)
                cma_es.tell(pop_solutions, erf_values)
                cma_es.disp()

        # Compute final ERF from selected indices
        cma_res = cma_es.result[0]
        selected_index = cma_res.astype(int)
        selected_modulo = selected_index % self.prior.shape[0]
        erf_cma = self.error_function(selected_modulo)
        return erf_cma, selected_modulo
