"""
Main file for Compressor
"""

# import os
# import shutil
# import logging
import argparse

# from compressor.pdf_grid import XGrid
# from compressor.pdf_grid import PdfSet
# from compressor.estimators import Estimators
# from compressor.estimators import NormalizationK


def nb_flavors(value):
    """
    Checks the nb of flavors from input
    """
    if value < 0 and value > 7:
        raise argparse.ArgumentTypeError(f"Value {value} is not allowed.")
    return value


def argument_parser():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="Compress PDF sets")
    parser.add_argument("-p", "--pdfset", help="PDF set", required=True)
    parser.add_argument(
        "-n", "--compress", help="Number of compressed replicas", required=True
    )
    parser.add_argument(
        "-f",
        "--nflavors",
        type=nb_flavors,
        help="Total number of flavors",
        required=True,
    )
    args = parser.parse_args()


def main():
    """
    Main function
    """
    pass
