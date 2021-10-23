#!/usr/bin/env python3

import json
import lhapdf
import argparse
from rich.console import Console
from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet
from pycompressor.errfunction import ErfComputation

console = Console()
lhapdf.setVerbosity(0)

NF = 4
Q0 = 1.65
EST_DIC = {
    "corr_estimators": ["correlation"],
    "stat_estimators": ["kolmogorov_smirnov"],
    "moment_estimators": ["mean", "stdev", "skewness", "kurtosis"],
}


def arg_parser():
    """Perse input argument"""

    parser = argparse.ArgumentParser(description="Compute the ERFs between two PDFs.")
    parser.add_argument(
        "-p",
        "--prior",
        type=str,
        help="Baseline PDF to be compared with.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cmpdf",
        type=str,
        help="PDF from which the ERFs will be computed.",
        required=True,
    )
    argument = parser.parse_args()
    return argument


def main():
    args = arg_parser()
    basename = args.prior
    cmpfname = args.cmpdf
    xgrid = XGrid().build_xgrid()

    # Compute PDF Grids
    pdfbase = PdfSet(basename, xgrid, Q0, NF).build_pdf()
    pdfcmpf = PdfSet(cmpfname, xgrid, Q0, NF).build_pdf()

    # Init. ERF class
    erfs = ErfComputation(
        pdfbase, EST_DIC, None, None, None, norm=False
    ).compute_all_erf(pdfcmpf)

    # Write results to file
    filename = f"ERFs_{basename}_vs_{cmpfname}.dat"
    outfile = open(filename, "w")
    outfile.write(json.dumps(erfs))
    outfile.close()
    console.print(f"Resutls stored in {filename}.", style="bold blue")


if __name__ == "__main__":
    main()
