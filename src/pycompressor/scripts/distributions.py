#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lhapdf
import logging
import pathlib
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from rich.logging import RichHandler
from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rich")

sns.set_style("whitegrid")


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


def plot_dists_per_fl(x, prior_fl, stand_fl, enhcd_fl, info, fit, folder):
    plotName = info["plotName"]
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(18, 18))
    xind = [5, 10, 15, 45, 55, 68]
    for i, axis in enumerate(axes.reshape(-1)):
        pprior = prior_fl[:, xind[i]]
        sstand = stand_fl[:, xind[i]]
        eenhcd = enhcd_fl[:, xind[i]]
        mp, sp = stats.norm.fit(pprior)
        ms, ss = stats.norm.fit(sstand)
        me, se = stats.norm.fit(eenhcd)
        _, binp, _ = axis.hist(
            pprior,
            histtype="step",
            bins=info.get("bins", 20),
            color="green",
            alpha=1,
            label="Prior",
            linewidth=2.5,
            density=True
        )

        # Prior
        pmin, pmax = min(binp), max(binp)
        lnp = np.linspace(pmin, pmax, len(pprior))
        pfit = stats.norm.pdf(lnp, mp, sp)  # Prior
        sfit = stats.norm.pdf(lnp, ms, ss)  # Standard
        efit = stats.norm.pdf(lnp, me, se)  # Enhanced
        # Compute KL ests
        kl_stand = stats.ks_2samp(pprior, sstand, mode="auto")
        kl_enhcd = stats.ks_2samp(pprior, eenhcd, mode="auto")

        axis.hist(
            sstand,
            histtype="step",
            bins=info.get("bins", 20),
            color="deeppink",
            alpha=1,
            label=f"Standard  KL(s={kl_stand[0]:.4f}, p={kl_stand[1]:.4f})",
            linewidth=2.5,
            density=True
        )
        axis.hist(
            eenhcd,
            histtype="step",
            bins=info.get("bins", 20),
            color="dodgerblue",
            alpha=1,
            label=f"Enhanced KL(s={kl_enhcd[0]:.4f}, p={kl_enhcd[1]:.4f})",
            linewidth=2.5,
            density=True
        )
        if fit:
            # Plot fits
            axis.plot(lnp, pfit, "--", color="green", linewidth=2)
            axis.plot(lnp, sfit, "--", color="deeppink", linewidth=2)
            axis.plot(lnp, efit, "--", color="dodgerblue", linewidth=2)
        axis.set_title(rf"$x$={x[xind[i]]:.4f}", fontsize=16)
        axis.tick_params(length=7, width=1.5)
        axis.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{folder}/{plotName}.png", dpi=250)
    plt.close("all")


def plot_dists(x, prior, stand, ehncd, folder, bins=10, fit=False):
    info = {
        "bins": bins,
        "xlabel": "x",
        "ylabel": "y",
    }
    for fl in range(prior.shape[0]):
        info["figTitle"] = IDS[fl]
        info["plotName"] = f"{IDS[fl]}"
        plot_dists_per_fl(x, prior[fl], stand[fl], ehncd[fl], info, fit, folder)


def arg_parser():
    """Parse inputs data file"""
    parser = argparse.ArgumentParser(description="Plot ERFs validation plot.")
    parser.add_argument("--pdf", help="PDF name", required=True)
    parser.add_argument("--cmpsize", help="Size of compressed PDF", required=True, type=posint)
    arguments = parser.parse_args()
    return arguments


def main():
    args = arg_parser()
    pdf_name = args.pdf
    comp_size = args.cmpsize

    # Get PDF names
    prior = str(pdf_name)
    comp_prior = f"{prior}_compressed_{comp_size}"
    comp_enhanced = f"{prior}_enhanced_compressed_{comp_size}"

    # Create output folder
    folder = pathlib.Path().absolute() / f"xDistributions_N{comp_size}"
    folder.mkdir(exist_ok=True)

    logger.info("Generating PDF grids.")
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(prior, xgrid, Q0, NF).build_pdf()
    cmprior = PdfSet(comp_prior, xgrid, Q0, NF).build_pdf()
    enhanced = PdfSet(comp_enhanced, xgrid, Q0, NF).build_pdf()

    # Transpose PDFs
    prior = np.transpose(prior, axes=[1, 0, 2])
    cmprior = np.transpose(cmprior, axes=[1, 0, 2])
    enhanced = np.transpose(enhanced, axes=[1, 0, 2])

    logger.info("Plot distributions.")
    plot_dists(xgrid, prior, cmprior, enhanced, folder)


if __name__ == "__main__":
    main()
