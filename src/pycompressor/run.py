"""
Main file for Compressor
"""

import os

# import logging
import json
import argparse
import numpy as np
from tqdm import trange

from pycompressor.pdf_grid import XGrid
from pycompressor.pdf_grid import PdfSet
from pycompressor.compressor import compress


def positive_int(value):
    """
    Checks if the given number is positive
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Negative values are not allowed: {ivalue}")
    return ivalue


def nbflavors(value):
    """
    Checks the nb of flavors from input
    """
    ivalue = int(value)
    if ivalue < 0 or ivalue > 7:
        raise argparse.ArgumentTypeError(f"Value {value} is not allowed.")
    return ivalue


def argument_parser():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="Compress PDF sets")
    parser.add_argument("-p", "--pdfset", help="Input PDF set", required=True)
    parser.add_argument("-q", "--qinit", type=positive_int, help="Scale Q0")
    parser.add_argument("-m", "--minimizer", help="Minimizer name")
    parser.add_argument(
        "-f", "--nflavors", type=nbflavors, help="Total number of flavors"
    )
    parser.add_argument(
        "-n",
        "--compress",
        type=positive_int,
        help="Number of compressed replicas",
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    """
    Main function
    """
    args = argument_parser()
    # Initi. parameters
    pdf = args.pdfset
    nbr = args.compress

    # Define default flavour number
    if args.nflavors is None:
        nfl = 3
    else:
        nfl = args.nflavors
    # Define default initial scale
    if args.qinit is None:
        q0 = 1
    else:
        q0 = args.qinit
    # Default Minimizer
    if args.minimizer is None:
        minimizer = "genetic"
    else:
        minimizer = args.minimizer

    # List of estimators
    est_dic = {
        "moment_estimators": ["mean", "stdev", "skewness", "kurtosis"],
        "stat_estimators": ["kolmogorov_smirnov"],
        "corr_estimators": ["correlation"],
    }

    # Construc xgrid
    print("[+] Loading PDF set:")
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(pdf, xgrid, q0, nfl).build_pdf()
    # Set seed
    np.random.seed(0)
    # Init. compressor class
    comp = compress(prior, est_dic, nbr)
    # Start compression depending on the Evolution Strategy
    erf_list = []
    final_result = {"pdfset_name": pdf}

    print(f"\n[+] Compressing replicas using {minimizer} algorithm:")
    if minimizer == "genetic":
        # Run compressor using GA
        nb_iter = 15000
        with trange(nb_iter) as iter_range:
            for i in iter_range:
                iter_range.set_description("Compression")
                erf, index = comp.genetic_algorithm(nb_mut=5)
                erf_list.append(erf)
                iter_range.set_postfix(ERF=erf)
    elif minimizer == "cma":
        # Run compressor using CMA
        erf, index = comp.cma_algorithm(std_dev=0.3)
        # erf, index = comp.cma_algorithm_multiprocessed(std_dev=0.3)
    else:
        raise ValueError(f"{minimizer} is not a valid minimizer.")

    # Create Output folders
    if not os.path.exists(pdf):
        os.mkdir(pdf)
    else:
        pass

    # Prepare output file
    final_result["ERFs"] = erf_list
    final_result["index"] = index.tolist()
    with open(
        f"{pdf}/compress_{args.pdfset}_{args.compress}_output.dat", "w"
    ) as outfile:
        json.dump(final_result, outfile)

    # Fetching ERF and construct reduced PDF grid
    print(f"\n[+] Final ERF: {erf}\n")

    # Compute final ERFs for the final choosen replicas
    final_err_func = comp.final_erfs(index)
    serfile = open("erf_reduced.dat", "a+")
    serfile.write(f"{nbr} ")
    for err in final_err_func.keys():
        serfile.write(f"{final_err_func[err]} ")
    serfile.write("\n")
    serfile.close()
