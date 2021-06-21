# Utils

import ast
import json
import logging
import numpy as np

log = logging.getLogger(__name__)


def preprocess_enhanced(enhanced, dec_check=15):
    """Pre-process the enhanced set by removing duplicates
    in the PDF grid.

    Parameters
    ----------
    enhanced: np.array(float)
        enhanced PDF grid

    Returns
    -------
    tuple(np.array, np.array, np.array)
        tuple that returns the pre-processed array, the indices
        that are kept and the number of times each array occured.
    """

    rounded = np.round(enhanced, dec_check)
    preprocessed, index, counts = np.unique(
            rounded,
            axis=0,
            return_index=True,
            return_counts=True
    )
    return preprocessed, index, counts


def map_index(refarr, arr):
    """Map the the elements in `arr` to the index in which
    they occur in `refarr`.

    Parameters
    ----------
    arr: np.array(int)
        one dimensional array of integers with size N
    refarr: np.array(int)
        one dimentional array of integers with size M

    Returns
    -------
    np.array(int)
        one dimentional array of integers with size N
    """

    inds = {e:i for i, e in enumerate(refarr)}
    return np.vectorize(inds.get)(arr)


def restore_permutation(index, shuffle, preprocess):
    """Undo the maping of indices due to the preprocessing
    and the shuffling.

    Parameters
    ----------
    index: np.array()
        array containing the final indices
    shuffle: np.array(float)
        array containing the permutation
    preprocess: np.array(float)
        array containing the indices of the pre-processing

    Returns
    -------
    np.array(float)
        array of index
    """

    undo_shuffle = shuffle[index]
    undo_preproc = preprocess[undo_shuffle]
    return undo_preproc


def remap_index(index, shuffled):
    new_idx = []
    for idx in index:
        # TODO: Implement exception
        pos = np.where(shuffled == idx)[0][0]
        new_idx.append(pos)
    return np.array(new_idx)


def extract_estvalues(comp_size):
    """Extract the result from the prior for a given
    compressed set (w.r.t the size).

    Parameters
    ----------
    comp_size : int
        Size of the compressed set
    """

    infile = "erfs_output/erf_reduced.dat"
    try:
        readfile = open(infile, "r").readlines()
        for line in readfile:
            infoline = line.strip("\n").split(":", 1)
            rep_size = int(infoline[0])
            if rep_size == comp_size:
                dic_estm = ast.literal_eval(infoline[1])
    except FileNotFoundError as err:
        log.critical(f"{err}")
    return dic_estm


def extract_index(pdfname, comp_size):
    """Extract the list of indices for a given compressed
    set (w.r.t the size)

    Parameters
    ----------
    pdfname: str
        Name of the original/input PDF
    comp_size : int
        Size of the compressed set
    """

    infile = f"{pdfname}/compress_{pdfname}_{comp_size}_output.dat"
    try:
        with open(infile) as results_file:
            results = json.load(results_file)
    except FileNotFoundError as err:
        log.critical(err)
    index = results["index"]
    return index


def extract_bestErf(pdfname, comp_size):
    """Extract the best/final ERF value for a given
    compressed set (w.r.t the size).

    Parameters
    ----------
    pdfname: str
        Name of the original/input PDF
    comp_size : int
        Size of the compressed set
    """

    infile = f"{pdfname}/compress_{pdfname}_{comp_size}_output.dat"
    try:
        with open(infile) as results_file:
            results = json.load(results_file)
    except FileNotFoundError as err:
        log.critical(err)
    bestErf = results["ERFs"]
    return bestErf[-1]


def compare_estimators(est1, est2):
    """Compare if the values of all the estimators in `est1` are
    samller or equal than in `est2` (`est1`<`est2`) and returns
    True if it is the case.

    Parameters
    ----------
    est1 :
        Value of the first estimator
    est2 :
        Value of the second estimator
    """

    if est2 is None:
        return True
    diffkeys = [k for k in est1 if est1[k] > est2[k]]
    return len(diffkeys) == 0


def get_best_estimator(list_ests):
    """Get the best estimator from a list of dictionaries
    containing values of all the different estimators.

    Parameters
    ----------
    list_ests: list
        List of dictionaries containing the results of
        all the statistical estimators
    """
    if len(list_ests) == 1:
        return list_ests[0]

    indx, best_est = 0, list_ests[0]

    for est in range(1, len(list_ests)):
        if compare_estimators(list_ests[est], best_est):
            indx, best_est = est, list_ests[est]
    return indx, best_est
