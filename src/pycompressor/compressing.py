import json
import shutil
import logging
import pathlib
import numpy as np
import subprocess as sub

from tqdm import trange
from pycompressor.pdfgrid import XGrid
from pycompressor.pdfgrid import PdfSet
from pycompressor.compressor import compress

log = logging.getLogger(__name__)

# Initial scale (in GeV)
Q0 = 1
# Total number of flavour to 2nf+1=7
NF = 3


def compressing(pdf, compressed, minimizer, est_dic, enhance, nbgen):
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

    if not enhance:
        pdf = str(pdf)
    else:
        from pycompressor.postgans import postgans
        # Enhance with GANs
        outfolder = str(pdf) + "_enhanced"
        sub.call(
            [
                "ganpdfs",
                "runcard.yml",
                "-o",
                f"{outfolder}",
                "-k",
                f"{nbgen}",
                "--force",
            ]
        )
        # Evolve Generated Grids
        shutil.copy("filter.yml", f"{outfolder}/filter.yml")
        sub.call(["evolven3fit", f"{outfolder}", f"{nbgen}"])
        # Add symbolic Links to LHAPDF dataDir
        postgans(str(pdf), outfolder, nbgen)
        pdf = str(pdf) + "_enhanced"

    # Create output folder
    folder = pathlib.Path().absolute() / pdf
    folder.mkdir(exist_ok=True)
    # Create output folder for ERF stats
    out_folder = pathlib.Path().absolute() / "erfs_output"
    out_folder.mkdir(exist_ok=True)
    log.info("Loading PDF set:")
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(pdf, xgrid, Q0, NF).build_pdf()
    # Set seed
    np.random.seed(0)

    # Init. compressor class
    comp = compress(prior, est_dic, compressed, out_folder)
    # Start compression depending on the Evolution Strategy
    erf_list = []
    final_result = {"pdfset_name": pdf}

    log.info(f"\nCompressing replicas using {minimizer} algorithm:")
    if minimizer == "genetic":
        # Run compressor using GA
        nb_iter = 15000
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
    with open(
        f"{pdf}/compress_{pdf}_{compressed}_output.dat", "w"
    ) as outfile:
        json.dump(final_result, outfile)

    # Fetching ERF and construct reduced PDF grid
    log.info(f"\nFinal ERF: {erf}\n")

    # Compute final ERFs for the final choosen replicas
    final_err_func = comp.final_erfs(index)
    serfile = open(f"{out_folder}/erf_reduced.dat", "a+")
    serfile.write(f"{compressed} ")
    for err in final_err_func.keys():
        serfile.write(f"{final_err_func[err]} ")
    serfile.write("\n")
    serfile.close()
