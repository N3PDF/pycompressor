"""
Main file for Compressor
"""

# import os
# import logging
import argparse

from pycompressor.pdf_grid import XGrid
from pycompressor.pdf_grid import PdfSet
from pycompressor.compressor import error_function


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

    # Construc xgrid
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(pdf, xgrid, q0, nfl).build_pdf()

    # Compute error function
    erf_tot = error_function(prior, nbr)
    print(f"Total ERF: {erf_tot}")
