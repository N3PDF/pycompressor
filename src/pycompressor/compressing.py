import json
import shutil
import logging
import pathlib
import numpy as np
import NNPDF as nnpath
import subprocess as sub

from tqdm import trange
from numpy.random import Generator, PCG64
from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet
from pycompressor.compressor import compress
from pycompressor.utils import remap_index
from pycompressor.utils import extract_index
from pycompressor.utils import extract_estvalues

log = logging.getLogger(__name__)

# Initial scale (in GeV)
Q0 = 1
# Total number of flavour to 2nf+1=7
NF = 3


class LoadingEnhancedError(Exception):
    """Compressor error."""
    pass


def splash():
    info = """\033[34m
+-------------------------------------------------------------------------+
|𝖕𝖞𝕮𝖔𝖒𝖕𝖗𝖊𝖘𝖘𝖔𝖗:                                                            |
|-------                                                                  |
|Fast python compressor for PDF replicas.                                 |
|https://n3pdf.github.io/pycompressor/                                    |
|© N3PDF                                                                  |
+-------------------------------------------------------------------------+ 
           """
    print(info + '\033[0m \033[97m')


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
        resultspath = nnpath.get_results_path()
        resultspath = resultspath + f"{str(pdf)}/filter.yml"
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
        # Evolve Generated Grids
        shutil.copy(resultspath, f"{outfolder}/filter.yml")
        sub.call(["evolven3fit", f"{outfolder}", f"{nbgen}"])
        # Add symbolic Links to LHAPDF dataDir
        postgans(str(pdf), outfolder, nbgen)

    splash()
    # Set seed
    rndgen = Generator(PCG64(seed=0))

    log.info("Loading PDF sets:")
    xgrid = XGrid().build_xgrid()
    # Load Prior Sets
    prior = PdfSet(pdf, xgrid, Q0, NF).build_pdf()
    rndindex = rndgen.integers(1, prior.shape[0], compressed + 1)
    # Load Enhanced Sets
    if enhanced_already_exists:
        try:
            postgan = pdf + "_enhanced"
            final_result = {"pdfset_name": postgan}
            enhanced = PdfSet(postgan, xgrid, Q0, NF).build_pdf()
            print(f"[+] Enhanced PDF set with {enhanced.shape[0]} loaded.")
        except RuntimeError as excp:
            raise LoadingEnhancedError(f"{excp}")
        nb_iter, ref_estimators = 30000, None
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
    log.info(f"Compressing replicas using {minimizer} algorithm:")
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
    log.info(f"Final ERF: {erf}\n")

    # Compute final ERFs for the final choosen replicas
    final_err_func = comp.final_erfs(index)
    serfile = open(f"{out_folder}/erf_reduced.dat", "a+")
    serfile.write(f"{compressed}:")
    serfile.write(json.dumps(final_err_func))
    serfile.write("\n")
    serfile.close()
