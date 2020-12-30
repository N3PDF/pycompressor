#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pylab
import lhapdf
import pathlib
import logging
import argparse
import numpy as np
import matplotlib.cm as cmx
import matplotlib.pyplot as plt

from numba import njit
from scipy.stats import spearmanr
from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet


logging.basicConfig(
        level=logging.INFO,
        format="\033[0;32m[%(levelname)s]\033[97m %(message)s"
    )
logger = logging.getLogger(__name__)


FLS = [
    [21,1],
    [21,2],
    [21,3],
    [1,2],
    [1,3],
    [2,3],
    [21,-1],
    [21,-2],
    [21,-3],
    [-1,-2],
    [-1,-3],
    [-2,-3],
    [1,-1],
    [2,-2],
    [3,-3],
    [1,-2],
    [-1,2],
    [-2,3],
    [2,-3],
    [1,-3],
    [-1,3]
]

TAGS = {
    21: "Gluon",
    1: "Down",
    2: "Up",
    3: "Strange",
    -1: "Antidown",
    -2: "Antiup",
    -3: "Antistrange"
}

XGRID = np.logspace(
    math.log(1e-5),
    math.log(0.9),
    num=100,
    base=math.exp(1)
)

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


def correlation(x, q, fl1, fl2, pdf, ind):
    """Compute correlations between two flavors for a
    given value of x. Exact copy of the function here:
    https://github.com/scarrazza/compressor/tree/master/tools

    Parameters:
    ----------
    x: float
        momentum fraction x
    q: float
        Energy scale
    fl1: int
        Flavor index
    fl2: int
        Flavor index
    pdf: lhapdf.mkPDF
        LHAPDF instance
    ind: list
        List of PDF indexes
    """
    nrep = len(ind)
    a = b = ab = sq_a = sq_b = 0.0
    for r in range(0, nrep):
        v1 = pdf[ind[r]].xfxQ(fl1, x, q)
        v2 = pdf[ind[r]].xfxQ(fl2, x, q)
        a  += v1
        b  += v2
        ab += v1 * v2
        sq_a += v1 * v1
        sq_b += v2 * v2
    a /= nrep
    b /= nrep
    ab /= nrep
    sig1 = math.sqrt(sq_a / nrep - a * a)
    sig2 = math.sqrt(sq_b / nrep - b * b)
    part1 = nrep / (nrep - 1.0)
    part2 = (ab - a * b) / (sig1 * sig2)
    return part1 * part2


def generate_pdf(pdfname):
    pdf, pdf_index = [], []
    pdfset = lhapdf.getPDFSet(f"{pdfname}")
    for i in range(1, pdfset.size):
        pdf_index.append(i)
    for i in range(0, pdfset.size):
        pdf.append(pdfset.mkPDF(i))
    return pdf, pdf_index


def onetime_pdf_generation(pdfname, xgrid, Nf=3, q0=1):
    init_pdf = PdfSet(pdfname, xgrid, q0, Nf) 
    init_pdf = init_pdf.build_pdf()
    pdf_indx = np.arange(init_pdf.shape[0])
    return init_pdf

def plot_correlation(x, prior, cmprior, enhanced, size, q=100):
    for fl in FLS:
        plt.figure(figsize=[10, 8])
        p_arr = np.empty(x.size)
        c_arr = np.empty(x.size)
        e_arr = np.empty(x.size)
        for i in range(0, x.size):
            p_arr[i] = correlation(x[i], q, fl[0], fl[1], prior[0], prior[1])
            c_arr[i] = correlation(x[i], q, fl[0], fl[1], cmprior[0], cmprior[1])
            e_arr[i] = correlation(x[i], q, fl[0], fl[1], enhanced[0], enhanced[1])
        plt.plot(x, p_arr, color='k', linewidth=2.0, label="Prior")
        plt.plot(x, c_arr, '--', color='b', linewidth=2.0, label="Standard")
        plt.plot(x, e_arr, '--', color='r', linewidth=2.0, label="Enhanced")

        plt.grid(True)
        plt.xlabel('x')
        plt.xscale('log')
        plt.xlim([x[0], x[-1]])
        plt.ylabel('Correlation')
        legend = plt.legend(loc=0, fancybox=True, framealpha=0.5)
        plt.title(f"{TAGS[fl[0]]}-{TAGS[fl[1]]} correlation (Compression to {size})")
        plt.savefig(f"correlations/{TAGS[fl[0]]}-{TAGS[fl[1]]}_correlation.png", dpi=350)
        plt.close()

@njit
def correlations(x1, x2, fl1, fl2, pdf, nrep):
    """Compute correlations between two flavors for a
    given value of x. Exact copy of the function here:
    https://github.com/scarrazza/compressor/tree/master/tools

    Parameters:
    ----------
    x1: float
        momentum fraction x
    x2: float
        momentum fraction x
    q: float
        Energy scale
    fl1: int
        Flavor index
    fl2: int
        Flavor index
    pdf: lhapdf.mkPDF
        LHAPDF instance
    ind: list
        List of PDF indexes
    """
    a = b = ab = sq_a = sq_b = 0.0
    for rp in range(nrep):
        v1 = pdf[rp][fl1][x1]
        v2 = pdf[rp][fl2][x2]
        a  += v1
        b  += v2
        ab += v1*v2
        sq_a += v1*v1
        sq_b += v2*v2
    a /= nrep
    b /= nrep
    ab /= nrep
    sig1 = np.sqrt(sq_a / (nrep - 1.0) - nrep / (nrep - 1.0) * a * a)
    sig2 = np.sqrt(sq_b / (nrep - 1.0) - nrep / (nrep - 1.0) * b * b)
    return nrep / (nrep - 1.0) * (ab - a * b) / (sig1 * sig2)


@njit
def compute_corrMatrix(x, pdfset):
    nrep, nflv, nxgd = pdfset.shape
    assert nxgd == x.size
    corr = np.zeros(shape=(nflv * nxgd, nflv * nxgd))
    for fl1 in range(nflv):
        for fl2 in range(nflv):
            for xi in range(nxgd):
                for xj in range(nxgd):
                    pi = fl1 * nxgd + xi
                    pj = fl2 * nxgd + xj
                    if (pj >= pi):                
                        # prior
                        corr[pi, pj] = correlations(
                            xi,
                            xj,
                            fl1,
                            fl2,
                            pdfset,
                            nrep
                        )
                        corr[pj, pi] = corr[pi, pj]
    return corr


def plot_corrMatrix(corr_mat, size, title=None, name=None):
    fig, axis = plt.subplots(figsize=[8, 8])
    plt.imshow(corr_mat, cmap='RdBu', vmin=-1, vmax=1)
    plt.title(title)
    plt.savefig(f"correlations/{name}.png", dpi=350)
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

    # Create output folder
    folder = pathlib.Path().absolute() / "correlations"
    folder.mkdir(exist_ok=True)

    # Get PDF names
    prior_name = str(pdf_name)
    cmprior_name = f"{prior_name}_compressed_{comp_size + 1}"
    enhanced_name = f"{prior_name}_enhanced_compressed_{comp_size + 1}"

    # Call PDFs
    logger.info("Generating PDFs.")
    # Usual calls
    prior = generate_pdf(prior_name)
    cmprior = generate_pdf(cmprior_name)
    enhanced = generate_pdf(enhanced_name)
    # Using pyCompressor calls. Generate the Grid ahead
    # in order to use Numba.
    mprior = onetime_pdf_generation(prior_name, XGRID)
    mcmprior = onetime_pdf_generation(cmprior_name, XGRID)
    menhanced = onetime_pdf_generation(enhanced_name, XGRID)

    # Correlation Plots
    logger.info("Plot correlations.")
    plot_correlation(XGRID, prior, cmprior, enhanced, comp_size)

    # Compute Correlation Matrix
    logger.info("Compute & plot correlation matrix.")
    corr_prior = compute_corrMatrix(XGRID, mprior)
    corr_stand = compute_corrMatrix(XGRID, mprior)
    corr_enhcd = compute_corrMatrix(XGRID, mprior)
    plot_corrMatrix(corr_prior, comp_size, title="Prior", name="prior")
    plot_corrMatrix(corr_stand, comp_size, title="Standard", name="standard")
    plot_corrMatrix(corr_enhcd, comp_size, title="Enhanced", name="enhanced")
