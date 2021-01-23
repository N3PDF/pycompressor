#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lhapdf
import pathlib
import logging
import argparse
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from rich.logging import RichHandler
from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet


logging.basicConfig(
        level=logging.INFO,
        format="\033[0;32m[%(levelname)s]\033[97m %(message)s",
        handlers=[RichHandler()]
    )
logger = logging.getLogger(__name__)


Q0 = 1  # Initial scale (in GeV)
NF = 3  # Total number of flavour to 2nf+1=7
IDS = [
    r"$\bar{s}$",
    r"$\bar{u}$",
    r"$\bar{d}$",
    r"$g$",
    r"$d$",
    r"$u$",
    r"$s$"
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


def fid(prior, compressed):
    """Similarity Metric Measure that measures the quality of the compressed PDF replicas
    using the `Fréchet Inception Distance` (FID).

    TODO: Check how the total/final FIDs is computed.

    Parameters
    ----------
    prior : np.array(float)
        Prior MC PDF replicas of shape (Nf, N, X)
    compressed : np.array(float)
        Compressed MC PDF replicas of shape (Nf, \tilde{N}, X)

    Returns
    -------
    float:
        FID
    """
    fid_arr = np.zeros(prior.shape[0])

    def compute_fid_per_fl(fl_prior, fl_compressed):
        """Measure the quality of the compressed PDF using the `Fréchet Inception Distance`
        (FID). The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        If the compressed PDF replica is exactly the same as the prior, the value of the FID 
        is zero; that means that the smaller the value of the FID is, the similar the compressed 
        replica is to the prior.

        For details about the FID's Inception Score measure, refer to the following:
        https://arxiv.org/abs/1706.08500

        Parameters
        ----------
        fl_prior : np.array(float)
            Array of prior PDF replica for a given flavor
        fl_compressed : np.array(float)
            Array of compressed PDF replica for a given flavor

        Returns
        -------
        float:
            FID value
        """
        # calculate mean and covariance statistics
        mu1, sigma1 = fl_prior.mean(axis=0), np.cov(fl_prior, rowvar=False)
        mu2, sigma2 = fl_compressed.mean(axis=0), np.cov(fl_compressed, rowvar=False)
        # Check if Infs or NaNs and return a big nnumber
        if (np.isnan(mu2).any() or np.isnan(sigma2).any()):
            return np.random.randint(400, 1000)

        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    for fl in range(prior.shape[0]):
        fid_arr[fl] = compute_fid_per_fl(prior[fl], compressed[fl])

    return fid_arr


def plot_hists(fid_rnd, fid_prior, fid_enhcd, info, width=.55):
    plotName = info["plotName"]
    x = np.arange(1, 2 * len(fid_rnd), step=2)
    fig, axis = plt.subplots(figsize=(14,8))
    # Plot bars
    r_bar = axis.bar(x - width, fid_rnd, width, label='Random')
    p_bar = axis.bar(x, fid_prior, width, label='Standard')
    e_bar = axis.bar(x + width, fid_enhcd, width, label='Enhanced')
    # Legends & Labels
    axis.set_xticks(x)
    axis.set_yscale("log")
    axis.set_xlabel(info["xlabel"])
    axis.set_ylabel(info["ylabel"])
    axis.set_title(info["plotTitle"])
    axis.grid(alpha=0.1, linewidth=1.5)
    axis.tick_params(length=7, width=1.5)
    axis.set_xticklabels(info["xticksLabels"])
    axis.legend()
    fig.tight_layout()
    plt.savefig(f"FIDs/{plotName}.png", dpi=250)
    plt.close("all")


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
    comp_prior = f"{prior}_compressed_{comp_size}"
    comp_enhanced = f"{prior}_enhanced_compressed_{comp_size}"

    # Create output folder
    folder = pathlib.Path().absolute() / f"FIDs"
    folder.mkdir(exist_ok=True)

    logger.info("Generateing PDF grids.")
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(prior, xgrid, Q0, NF).build_pdf()
    cmprior = PdfSet(comp_prior, xgrid, Q0, NF).build_pdf()
    enhanced = PdfSet(comp_enhanced, xgrid, Q0, NF).build_pdf()

    # Select Random replicas for accuracy ref.
    indices = np.random.choice(prior.shape[0], cmprior.shape[0], replace=False)
    rndprior = prior[indices]

    # Transpose PDFs
    prior = np.transpose(prior, axes=[1, 0, 2])
    cmprior = np.transpose(cmprior, axes=[1, 0, 2])
    rndprior = np.transpose(rndprior, axes=[1, 0, 2])
    enhanced = np.transpose(enhanced, axes=[1, 0, 2])

    # FIDs
    logger.info("Generate FID plots.")
    # Compute FIDs
    fid_prior = fid(prior, cmprior)
    fid_rprior = fid(prior, rndprior)
    fid_enhanced = fid(prior, enhanced)
    # Plot info
    fid_info = {
        # Plot info
        "xlabel": "Flavours",
        "ylabel": "FID Values",
        "xticksLabels": IDS,
        "plotTitle": "Frechet Inception Distance (FID)",
        "plotName": f"FID_N{comp_size}"
    }
    plot_hists(fid_rprior, fid_prior, fid_enhanced, fid_info)
