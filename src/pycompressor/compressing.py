import json
import logging
import pathlib
import numpy as np
import subprocess as sub

from tqdm import trange
from rich.table import Table
from rich.style import Style
from rich.console import Console
from numpy.random import Generator, PCG64
from reportengine.checks import CheckError
from reportengine.checks import make_argcheck

from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet
from pycompressor.compressor import Compress
from pycompressor.utils import extract_index
from pycompressor.estimators import ALLOWED_ESTIMATORS

console = Console()
log = logging.getLogger(__name__)

# Initial scale (in GeV)
Q0 = 1
# Total number of flavour to 2nf+1=7
NF = 3


class LoadingEnhancedError(Exception):
    """Compressor error."""
    pass


def splash():
    """Splash information."""

    style = Style(color="blue")
    logo = Table(show_header=True, header_style="bold blue", style=style)
    logo.add_column("𝖕𝖞𝕮𝖔𝖒𝖕𝖗𝖊𝖘𝖘𝖔𝖗", justify="center", width=60)
    logo.add_row("[bold blue]Fast python compressor for PDF replicas.")
    logo.add_row("[bold blue]https://n3pdf.github.io/pycompressor/")
    logo.add_row("[bold blue]© N3PDF 2021")
    logo.add_row("[bold blue]Authors: Stefano Carrazza, Juan E. Cruz-Martinez, Tanjona R. Rabemananjara")
    console.print(logo)


@make_argcheck
def check_validity(pdfsetting, compressed, gans, est_dic):
    """ Check whether various quantities are acceptable """
    if not isinstance(compressed, int):
        raise CheckError(f"The key 'compressed' must be an integer, received: {compressed}")
    members = pdfsetting["pdf"].load().GetMembers()
    for estimator_list in est_dic.values():
        for estimator in estimator_list:
            if estimator not in ALLOWED_ESTIMATORS:
                raise CheckError(f"Estimator {estimator} is not allowed, allowed values are"
                        f"{ALLOWED_ESTIMATORS}")
    if members < compressed:
        if not gans["enhance"] and not pdfsetting["existing_enhanced"]:
            raise CheckError(
                    f" Cannot get {compressed} replicas from"
                    f" {members} members if enhancing is not active.")


@check_validity
def compressing(pdfsetting, compressed, minimizer, est_dic, gans):
    """
    Action that performs the compression. The parameters
    for the compression are provided by a `runcard.yml`.

    Parameters
    ----------
    pdf: str
        pdf/PDF name
    compressed: int
        Size of the compressed set
    est_dic: dict
        Dictionary containing the list of estimators
    """

    pdf = str(pdfsetting["pdf"])
    enhd_exists = pdfsetting.get("existing_enhanced", False)

    if gans["enhance"]:
        from pycompressor.postgans import postgans
        runcard = gans["runcard"]
        nbgen = gans["total_replicas"]
        # Write PDF name into gans runcard
        ganruncard = open(f"{runcard}.yml", "a+")
        ganruncard.write(f"pdf: {str(pdf)}")
        ganruncard.close()
        outfolder = str(pdf) + "_enhanced"
        sub.call(
            [
                "ganpdfs",
                f"{runcard}.yml",
                "-o",
                f"{outfolder}",
                "-k",
                f"{nbgen}",
                "--force",
            ]
        )
        sub.call(["evolven3fit", f"{outfolder}", f"{nbgen}"])
        # Add symbolic Links to LHAPDF dataDir
        postgans(str(pdf), outfolder, nbgen)

    splash()
    rndgen = Generator(PCG64(seed=0))

    xgrid = XGrid().build_xgrid()
    # Load Prior Sets
    prior = PdfSet(pdf, xgrid, Q0, NF).build_pdf()
    rndindex = rndgen.choice(prior.shape[0], compressed, replace=False)

    outname = [pdf]
    final_result = [{"pdfset_name": pdf}]
    nb_iter =[15000]
    ref_estimators = [None]
    init_index = [rndindex]
    enhanced = [prior]

    # Methodological iterations
    mtd_iteration = 2 if enhd_exists else 1

    for cmtype in range(mtd_iteration):
        # necessary to get the same normalization
        rndgen = Generator(PCG64(seed=0))
        _ = rndgen.choice(prior.shape[0], compressed, replace=False)
        # reference log
        if cmtype==0:
            console.print(
                    "Standard compression using Input set",
                    style="bold green underline"
            )
        elif cmtype==1:
            console.print(
                    "Adiabatic compression using Enhanced set",
                    style="bold green underline"
            )

        # Create output folder
        outrslt = outname[cmtype]
        folder = pathlib.Path().absolute() / outrslt
        folder.mkdir(exist_ok=True)
        # Create output folder for ERF stats
        out_folder = pathlib.Path().absolute() / "erfs_output"
        out_folder.mkdir(exist_ok=True)

        # Output Summary
        console.print("\n• Compression Summary:", style="bold blue")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Parameters", justify="left", width=24)
        table.add_column("Description", justify="left", width=50)
        table.add_row("PDF set name", f"{pdf}")
        table.add_row("Size of Prior", f"{prior.shape[0] - 1} replicas")
        if cmtype!=0 and enhd_exists:
            table.add_row(
                    "Size of enhanced",
                    f"{enhanced[1].shape[0] - 1} replicas"
            )
        table.add_row("Size of compression", f"{compressed} replicas")
        table.add_row("Input energy Q0", f"{Q0} GeV")
        table.add_row(
            "x-grid size",
            f"{xgrid.shape[0]} points, x=({xgrid[0]:.4e}, {xgrid[-1]:.4e})"
        )
        table.add_row("Minimizer", f"{minimizer}")
        console.print(table)

        # Init. Compressor class
        comp = Compress(
            prior,
            enhanced[cmtype],
            est_dic,
            compressed,
            init_index[cmtype],
            ref_estimators[cmtype],
            out_folder,
            rndgen
        )
        # Start compression depending on the Evolution Strategy
        erf_list = []
        console.print("\n• Compressing MC PDF replicas:", style="bold blue")
        if minimizer == "genetic":
            # Run compressor using GA
            with trange(nb_iter[cmtype]) as iter_range:
                for _ in iter_range:
                    iter_range.set_description("Compression")
                    erf, index = comp.genetic_algorithm(nb_mut=5)
                    erf_list.append(erf)
                    iter_range.set_postfix(ERF=erf)
        elif minimizer == "cma":
            # Run compressor using CMA
            erf, index = comp.cma_algorithm(std_dev=0.8)
        else:
            raise ValueError(f"{minimizer} is not a valid minimizer.")

        # Prepare output file
        final_result[cmtype]["ERFs"] = erf_list
        final_result[cmtype]["index"] = index.tolist()
        outfile = open(f"{outrslt}/compress_{pdf}_{compressed}_output.dat", "w")
        outfile.write(json.dumps(final_result[cmtype], indent=2))
        outfile.close()
        # Fetching ERF and construct reduced PDF grid
        console.print(f"\n• Final ERF: {erf}.", style="bold blue")

        if (cmtype!=0 and enhd_exists) or (cmtype==0 and not enhd_exists):
            # Compute final ERFs for the final choosen replicas
            final_err_func = comp.final_erfs(index)
            serfile = open(f"{out_folder}/erf_reduced.dat", "a+")
            serfile.write(f"{compressed}:")
            serfile.write(json.dumps(final_err_func))
            serfile.write("\n")
            serfile.close()

        # Load Enhanced Sets
        if cmtype==0 and enhd_exists:
            try:
                postgan = pdf + "_enhanced"
                outname.append(postgan)
                final_result.append({"pdfset_name": postgan})
                enhncd = PdfSet(postgan, xgrid, Q0, NF).build_pdf()
                enhanced.append(enhncd)
            except RuntimeError as excp:
                raise LoadingEnhancedError(f"{excp}")
            nb_iter.append(100000)
            ref_estimators.append(None)
            pre_index = np.array(extract_index(pdf, compressed))
            init_index.append(pre_index)
