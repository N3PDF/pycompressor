import json
import shutil
import logging
import pathlib
import numpy as np
import subprocess as sub

from tqdm import trange
from rich.table import Table
from rich.style import Style
from rich.console import Console
from numpy.random import Generator, PCG64
from reportengine.checks import CheckError, make_argcheck

from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet
from pycompressor.compressor import compress
from pycompressor.utils import remap_index
from pycompressor.utils import extract_index
from pycompressor.utils import extract_estvalues

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
def check_validity(pdfsetting, gans):
    # How many members has this PDF? Can we get enough replicas from it?
    members = pdfsetting["pdf"].get_members()
    requested_replicas = gans["total_replicas"]
    if members < requested_replicas:
        # Only valid if number if in enhancing mode or if enhanced exists
        if not gans["enhance"] and not pdfsetting["existing_enhanced"]:
            raise CheckError(f"Cannot get {requested_replicas} replicas from {members} members if enhancing is not active")


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
    enhanced_already_exists = pdfsetting.get("existing_enhanced", False)

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
    # Set seed
    rndgen = Generator(PCG64(seed=0))

    console.print("\n• Load PDF sets & Printing Summary:", style="bold blue")
    xgrid = XGrid().build_xgrid()
    # Load Prior Sets
    prior = PdfSet(pdf, xgrid, Q0, NF).build_pdf()
    rndindex = rndgen.choice(prior.shape[0], compressed, replace=False)
    # Load Enhanced Sets
    if enhanced_already_exists:
        try:
            postgan = pdf + "_enhanced"
            final_result = {"pdfset_name": postgan}
            enhanced = PdfSet(postgan, xgrid, Q0, NF).build_pdf()
        except RuntimeError as excp:
            raise LoadingEnhancedError(f"{excp}")
        nb_iter, ref_estimators = 25000, None
        init_index = np.array(extract_index(pdf, compressed))
    else:
        final_result = {"pdfset_name": pdf}
        nb_iter, ref_estimators = 15000, None
        init_index, enhanced = rndindex, prior

    # Create output folder
    outrslt = postgan if enhanced_already_exists else pdf
    folder = pathlib.Path().absolute() / outrslt
    folder.mkdir(exist_ok=True)
    # Create output folder for ERF stats
    out_folder = pathlib.Path().absolute() / "erfs_output"
    out_folder.mkdir(exist_ok=True)

    # Output Summary
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameters", justify="left", width=24)
    table.add_column("Description", justify="left", width=50)
    table.add_row("PDF set name", f"{pdf}")
    table.add_row("Size of Prior", f"{prior.shape[0] - 1} replicas")
    if enhanced_already_exists:
        table.add_row("Size of enhanced", f"{enhanced.shape[0] - 1} replicas")
    table.add_row("Size of compression", f"{compressed} replicas")
    table.add_row("Input energy Q0", f"{Q0} GeV")
    table.add_row(
        "x-grid size",
        f"{xgrid.shape[0]} points, x=({xgrid[0]:.4e}, {xgrid[-1]:.4e})"
    )
    table.add_row("Minimizer", f"{minimizer}")
    console.print(table)

    # Init. Compressor class
    comp = compress(
        prior,
        enhanced,
        est_dic,
        compressed,
        init_index,
        ref_estimators,
        out_folder,
        rndgen
    )
    # Start compression depending on the Evolution Strategy
    erf_list = []
    console.print("\n• Compressing MC PDF replicas:", style="bold blue")
    if minimizer == "genetic":
        # Run compressor using GA
        with trange(nb_iter) as iter_range:
            for i in iter_range:
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
    final_result["ERFs"] = erf_list
    final_result["index"] = index.tolist()
    outfile = open(f"{outrslt}/compress_{pdf}_{compressed}_output.dat", "w")
    outfile.write(json.dumps(final_result, indent=2))
    outfile.close()
    # Fetching ERF and construct reduced PDF grid
    console.print(f"\n• Final ERF: [bold red]{erf}.", style="bold red")

    # Compute final ERFs for the final choosen replicas
    final_err_func = comp.final_erfs(index)
    serfile = open(f"{out_folder}/erf_reduced.dat", "a+")
    serfile.write(f"{compressed}:")
    serfile.write(json.dumps(final_err_func))
    serfile.write("\n")
    serfile.close()
