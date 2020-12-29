#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pylab
import lhapdf
import logging
import argparse
import numpy as np
import matplotlib.cm as cmx
import matplotlib.pyplot as plt


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
    cmprior_name = f"{prior_name}_compressed_{comp_size + 1}"
    enhanced_name = f"{prior_name}_enhanced_compressed_{comp_size + 1}"
    # Call PDFs
    logger.info("Generating PDFs.")
    prior = generate_pdf(prior_name)
    cmprior = generate_pdf(cmprior_name)
    enhanced = generate_pdf(enhanced_name)

    # Actual plotting
    logger.info("Plot resutls.")
    plot_correlation(XGRID, prior, cmprior, enhanced, comp_size)
