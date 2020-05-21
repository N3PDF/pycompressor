"""
Main file for Compressor
"""

import os
import shutil
import logging
import argparse

from compressor.pdfGrid import xgrid
from compressor.pdfGrid import pdfSet
from compressor.estimators import estimators
from compressor.estimators import normalizationK


def nbFlavors(value):
    """
    Checks the nb of flavors from input
    """
    if value<0 and 7<value:
        raise argparse.ArgumentTypeError(f'Value {value} is not allowed.')
    return value


def argument_parser():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description="Compress PDF sets")
    parser.add_argument("-p",  "--pdfset", help="PDF set", required=True)
    parser.add_argument("-n",  "--compress", help="Number of compressed replicas", required=True)
    parser.add_argument("-f",  "--nflavors", type=nbFlavors, help="Total number of flavors", required=True)
    args = parser.parse_args()



def main():
    """
    Main function
    """
    pass


