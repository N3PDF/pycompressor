#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import lhapdf
import pathlib
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from matplotlib.ticker import *
from rich.logging import RichHandler
from pycompressor.pdfgrid import PdfSet


logging.basicConfig(
        level=logging.INFO,
        format="\033[0;32m[%(levelname)s]\033[97m %(message)s",
        handlers=[RichHandler()]
    )
logger = logging.getLogger(__name__)


FLS = [
    [21, 1],
    [21, 2],
    [21, 3],
    [1, 2],
    [1, 3],
    [2, 3],
    [21, -1],
    [21, -2],
    [21, -3],
    [-1, -2],
    [-1, -3],
    [-2, -3],
    [1, -1],
    [2, -2],
    [3, -3],
    [1, -2],
    [-1, 2],
    [-2, 3],
    [2, -3],
    [1, -3],
    [-1, 3]
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

X_FORMATTER = [
    '\t\t' + r'$\bar{s}$',
    '\t\t' + r'$\bar{u}$',
    '\t\t' + r'$\bar{d}$',
    '\t\t$g$',
    '\t\t$d$',
    '\t\t$u$',
    '\t\t$s$'
]

Y_FORMATTER = [
    '\n\n' + r'$\bar{s}$',
    '\n\n' + r'$\bar{u}$',
    '\n\n' + r'$\bar{d}$',
    '\n\n$g$',
    '\n\n$d$',
    '\n\n$u$',
    '\n\n$s$'
]

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
        a += v1
        b += v2
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


def generate_random_pdf(pdfname, compression_size):
    pdf, pdf_index = [], []
    pdfset = lhapdf.getPDFSet(f"{pdfname}")
    for i in range(1, compression_size):
        pdf_index.append(i)
    indices = np.random.choice(
        pdfset.size,
        compression_size,
        replace=False
    )
    for i in range(0, compression_size):
        pdf.append(pdfset.mkPDF(indices[i]))
    return pdf, pdf_index


def onetime_pdf_generation(pdfname, xgrid, Nf=3, q0=100):
    init_pdf = PdfSet(pdfname, xgrid, q0, Nf)
    init_pdf = init_pdf.build_pdf()
    # pdf_indx = np.arange(init_pdf.shape[0])
    return init_pdf


def plot_correlation(x, prior, random, cmprior, enhanced, size, folder, q=100):
    subfolder = folder / "PDF-Correlations"
    subfolder.mkdir(exist_ok=True)

    for fl in FLS:
        plt.figure(figsize=[10, 8])
        p_arr = np.empty(x.size)
        r_arr = np.empty(x.size)
        c_arr = np.empty(x.size)
        e_arr = np.empty(x.size)
        for i in range(0, x.size):
            p_arr[i] = correlation(x[i], q, fl[0], fl[1], prior[0], prior[1])
            r_arr[i] = correlation(x[i], q, fl[0], fl[1], random[0], random[1])
            c_arr[i] = correlation(x[i], q, fl[0], fl[1], cmprior[0], cmprior[1])
            e_arr[i] = correlation(x[i], q, fl[0], fl[1], enhanced[0], enhanced[1])
        plt.plot(x, p_arr, color='k', linewidth=2.0, label="Prior")
        plt.plot(x, r_arr, '--', color='g', linewidth=2.0, label="Random")
        plt.plot(x, c_arr, '--', color='b', linewidth=2.0, label="Standard")
        plt.plot(x, e_arr, '--', color='r', linewidth=2.0, label="Enhanced")

        plt.grid(True)
        plt.xlabel('x')
        plt.xscale('log')
        plt.xlim([x[0], x[-1]])
        plt.ylabel('Correlation')
        legend = plt.legend(loc=0, fancybox=True, framealpha=0.5)
        plt.title(f"{TAGS[fl[0]]}-{TAGS[fl[1]]} correlation (Compression to {size})")
        plt.savefig(f"{subfolder}/{TAGS[fl[0]]}-{TAGS[fl[1]]}_correlation.png", dpi=350)
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


def plot_corrMatrix(corr_mat, size, folder, v=[-1, 1], title=None, name=None):
    subfolder = folder / "CorrMatrix"
    subfolder.mkdir(exist_ok=True)

    fig, axis = plt.subplots(figsize=[8, 8])
    plt.imshow(corr_mat, cmap='RdBu', vmin=v[0], vmax=v[1])
    plt.title(title)
    plt.colorbar()
    frame = plt.gca()
    frame.axes.get_yaxis().set_major_locator(LinearLocator(8))
    frame.get_yaxis().set_major_formatter(FixedFormatter(Y_FORMATTER))
    frame.axes.get_xaxis().set_major_locator(LinearLocator(8))
    frame.get_xaxis().set_major_formatter(FixedFormatter(X_FORMATTER))
    plt.savefig(f"{subfolder}/{name}.png", dpi=350)
    plt.close("all")


def plot_DiffCorrMat(corr_stand, corr_enhcd, folder, name=None):
    subfolder = folder / "DiffCorrMatrix"
    subfolder.mkdir(exist_ok=True)

    corr_stand = corr_stand.flatten()
    corr_enhcd = corr_enhcd.flatten()
    p_rsd = corr_stand.std() / corr_stand.mean()
    e_rsd = corr_enhcd.std() / corr_enhcd.mean()
    fig, axis = plt.subplots(figsize=[10, 8])
    plt.hist(
        corr_stand,
        bins=18,
        linewidth=2,
        histtype="step",
        label="Prior-Standard"
    )
    plt.hist(
        corr_enhcd,
        bins=18,
        linewidth=2,
        histtype="step",
        label="Prior-Enhanced"
    )
    plt.legend(fontsize=16)
    plt.grid(alpha=0.45)
    standtitle = f"mean={corr_stand.mean():.4f}, std={corr_stand.std():.4f}"
    enchdtitle = f"mean={corr_enhcd.mean():.4f}, std={corr_enhcd.std():.4f}"
    plt.title(f"Prior-Standard : {standtitle}\nPrior-Enhanced: {enchdtitle}", fontsize=18)
    plt.savefig(f"{subfolder}/{name}.png", dpi=350)
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
    prior_name = str(pdf_name)
    cmprior_name = f"{prior_name}_compressed_{comp_size}"
    enhanced_name = f"{prior_name}_enhanced_compressed_{comp_size}"

    # Create output folder
    folder = pathlib.Path().absolute() / f"correlations_N{comp_size}"
    folder.mkdir(exist_ok=True)

    # Call PDFs
    logger.info("Generating PDFs.")
    # Usual calls
    prior = generate_pdf(prior_name)
    cmprior = generate_pdf(cmprior_name)
    enhanced = generate_pdf(enhanced_name)
    rndprior = generate_random_pdf(prior_name, comp_size)
    # Using pyCompressor calls. Generate the Grid ahead
    # in order to use Numba.
    mprior = onetime_pdf_generation(prior_name, XGRID)
    mcmprior = onetime_pdf_generation(cmprior_name, XGRID)
    menhanced = onetime_pdf_generation(enhanced_name, XGRID)

    # Correlation Plots
    logger.info("Plot correlations.")
    plot_correlation(
        XGRID,
        prior,
        rndprior,
        cmprior,
        enhanced,
        comp_size,
        folder
    )

    # Compute Correlation Matrix
    logger.info("Compute & plot correlation matrix.")
    corr_prior = compute_corrMatrix(XGRID, mprior)
    corr_stand = compute_corrMatrix(XGRID, mcmprior)
    corr_enhcd = compute_corrMatrix(XGRID, menhanced)
    plot_corrMatrix(corr_prior, comp_size, folder, title="Prior", name="prior")
    plot_corrMatrix(corr_stand, comp_size, folder, title="Standard", name="standard")
    plot_corrMatrix(corr_enhcd, comp_size, folder, title="Enhanced", name="enhanced")

    # DIfference
    prior_vs_stand = corr_stand - corr_prior
    prior_vs_enhcd = corr_enhcd - corr_prior
    plot_corrMatrix(
        prior_vs_stand,
        comp_size,
        folder,
        v=[-2e-1, 2e-1],
        title="Prior-Standard",
        name="P_vs_S"
    )
    plot_corrMatrix(
        prior_vs_enhcd,
        comp_size,
        folder,
        v=[-2e-1, 2e-1],
        title="Prior-Enhanced",
        name="P_vs_E"
    )

    logger.info("Project heatmaps into histograms.")
    plot_DiffCorrMat(prior_vs_stand, prior_vs_enhcd, folder, name="hist_project")
