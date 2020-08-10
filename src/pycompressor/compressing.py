import json
import logging
import pathlib
import numpy as np
from tqdm import trange

from pycompressor.pdf_grid import XGrid
from pycompressor.pdf_grid import PdfSet
from pycompressor.compressor import compress

log = logging.getLogger(__name__)

# Initial scale (in GeV)
Q0 = 1
# Total number of flavour to 2nf+1=7
NF = 3


def compressing(fit, compressed, minimizer, est_dic):
    """
    Action that performs the compression. The parameters
    for the compression are provided by a `runcard.yml`.

    Parameters
    ----------
        fit: str
            Fit/PDF name
        compressed: int
            Size of the compressed set
        est_dic: dict
            Dictionary containing the list of estimators
    """

    pdf = str(fit)
    # Create output folder
    folder = pathlib.Path().absolute() / pdf
    folder.mkdir(exist_ok=True)
    print("[+] Loading PDF set:")
    xgrid = XGrid().build_xgrid()
    prior = PdfSet(pdf, xgrid, Q0, NF).build_pdf()
    # Set seed
    np.random.seed(0)
    # Init. compressor class
    comp = compress(prior, est_dic, compressed)
    # Start compression depending on the Evolution Strategy
    erf_list = []
    final_result = {"pdfset_name": pdf}

    print(f"\n[+] Compressing replicas using {minimizer} algorithm:")
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
    print(f"\n[+] Final ERF: {erf}\n")

    # Compute final ERFs for the final choosen replicas
    final_err_func = comp.final_erfs(index)
    serfile = open("erf_reduced.dat", "a+")
    serfile.write(f"{compressed} ")
    for err in final_err_func.keys():
        serfile.write(f"{final_err_func[err]} ")
    serfile.write("\n")
    serfile.close()
