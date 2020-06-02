"""
Main file for Compressor
"""

# import os
# import logging
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
    parser.add_argument("-q", "--qinit", type=positive_int, help="Initial Scale Q0")
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

    if args.nflavors is None:
        nfl = 3
    else:
        nfl = args.nflavors

    if args.qinit is None:
        q0 = 1
    else:
        q0 = args.qinit

    # List of estimators
    est_dic = {
            'moment_estimators': [
                'mean',
                'stdev',
                'skewness',
                'kurtosis'
                ],
            'stat_estimators': [
                'kolmogorov_smirnov'
                ]
            }

    # GA parameters
    ga_params = {
            'number_mutation': 5
            }

    # Construc xgrid
    print('[+] Loading PDF set:')
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(pdf, xgrid, q0, nfl).build_pdf()

    # Set seed
    np.random.seed(0)

    # Init. compressor class
    comp = compress(prior, est_dic, nbr)

    # Run compressor
    nb_iter = 15000
    print('\n[+] Compressing replicas:')
    with trange(nb_iter) as iter_range:
        for i in iter_range:
            iter_range.set_description("Compression")
            erf, index = comp.genetic_algorithm(ga_params)
            iter_range.set_postfix(ERF=erf)

    # Fetching ERF and construct reduced PDF grid
    print(f'\n[+] Final ERF:{erf}\n')
