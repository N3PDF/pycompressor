#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lhapdf
import logging
import pathlib
import argparse
import numpy as np
from numba import njit
from scipy import interpolate
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet


logging.basicConfig(
        level=logging.INFO,
        format="\033[0;32m[%(levelname)s]\033[97m %(message)s"
    )
logger = logging.getLogger(__name__)


Q0 = 1  # Initial scale (in GeV)
NF = 3  # Total number of flavour to 2nf+1=7
IDS = [
    r"sbar",
    r"ubar",
    r"dbar",
    r"g",
    r"d",
    r"u",
    r"s"
]


np.random.seed(0)
lhapdf.setVerbosity(0)


def posint(value):
    """Checks that a given number is positive.
    """

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"Negative values are not allowed, received: {value}"
        )
    return ivalue


def plot_dists_per_fl(x, prior_fl, stand_fl, enhcd_fl, info):
    plotName = info["plotName"]
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(20, 18))
    xind = [5, 10, 15, 45, 55, 68]
    for i, axis in enumerate(axes.reshape(-1)):
        axis.hist(
            prior_fl[:,xind[i]],
            histtype="step",
            bins=info.get("bins", 20),
            color="green",
            alpha=1,
            label="Prior",
            linewidth=2.5,
            density=True
        )
        axis.hist(
            stand_fl[:,xind[i]],
            histtype="step",
            bins=info.get("bins", 20),
            color="deeppink",
            alpha=1,
            label="Standard",
            linewidth=2.5,
            density=True
        )
        axis.hist(
            enhcd_fl[:,xind[i]],
            histtype="step",
            bins=info.get("bins", 20),
            color="dodgerblue",
            alpha=1,
            label="Enhanced",
            linewidth=2.5,
            density=True
        )
        # Params & Info
        # axis.set_xlabel(info["xlabel"])
        # axis.set_ylabel(info["ylabel"])
        axis.set_title(f"x={x[xind[i]]}")
        axis.grid(alpha=0.1, linewidth=1.5)
        axis.tick_params(length=7, width=1.5)
        axis.legend()
    fig.suptitle(info["figTitle"])
    fig.savefig(f"distributions/{plotName}.png", dpi=250)
    plt.close("all")


def plot_dists(x, prior, stand, ehncd, bins=10):
    info = {
        "bins": bins,
        "xlabel": "x",
        "ylabel": "y",
    }
    folder = pathlib.Path().absolute() / "distributions"
    folder.mkdir(exist_ok=True)
    for fl in range(prior.shape[0]):
        info["figTitle"] = IDS[fl]
        info["plotName"] = f"dist_{IDS[fl]}"
        plot_dists_per_fl(x, prior[fl], stand[fl], ehncd[fl], info)


def arg_parser():
    """Parse inputs data file"""
    parser = argparse.ArgumentParser(description="Plot ERFs validation plot.")
    parser.add_argument("--pdf", help="PDF name", required=True)
    parser.add_argument("--cmpsize", help="Size of compressed PDF", required=True, type=posint)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = arg_parser()
    pdf_name = args.pdf
    comp_size = args.cmpsize

    # Get PDF names
    prior = str(pdf_name)
    comp_prior = f"{prior}_compressed_{comp_size + 1}"
    comp_enhanced = f"{prior}_enhanced_compressed_{comp_size + 1}"

    logger.info("Generateing PDF grids.")
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(prior, xgrid, Q0, NF).build_pdf()
    cmprior = PdfSet(comp_prior, xgrid, Q0, NF).build_pdf()
    enhanced = PdfSet(comp_enhanced, xgrid, Q0, NF).build_pdf()

    # Transpose PDFs
    prior = np.transpose(prior, axes=[1, 0, 2])
    cmprior = np.transpose(cmprior, axes=[1, 0, 2])
    enhanced = np.transpose(enhanced, axes=[1, 0, 2])

    logger.info("Plot distributions.")
    plot_dists(xgrid, prior, cmprior, enhanced)
