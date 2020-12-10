# Utils

import json
import logging

log = logging.getLogger(__name__)


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
                dic_estm = eval(infoline[1])
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
        log.critical(f"{err}")
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
        log.critical(f"{err}")
    bestErf = results["ERFs"]
    return bestErf[-1]


def compare_estimators(est1, est2):
    """Compare if the values of all the estimators in `est1` are
    samller or equal than in `est2` (`est1`<`est2`) and returns
    True if it is the case.

    Parameters
    ----------
    est1 :
        est1
    est2 :
        est2
    """
    diffkeys = [k for k in est1 if est1[k] > est2[k]]
    return len(diffkeys) == 0


def get_best_estimator(list_ests):
    """get_best_estimator.

    Parameters
    ----------
    list_ests :
        list_ests
    """
    if len(list_ests) == 1:
        return list_ests[0]
    else:
        indx, best_est = 0, list_ests[0]

    for est in range(1, len(list_ests)):
        if compare_estimators(list_ests[est], best_est):
            indx, best_est = est, list_ests[est]
    return indx, best_est
