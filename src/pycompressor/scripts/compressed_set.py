#!/usr/bin/env python3

import os
import sys
import json
import shutil
import lhapdf
import argparse
from tqdm import trange
from subprocess import PIPE
from subprocess import Popen
from rich.table import Table
from rich.console import Console

console = Console()
lhapdf.setVerbosity(0)


def arg_parser():
    """Perse input argument"""

    parser = argparse.ArgumentParser(description="Generate compressed grid.")
    parser.add_argument("-i", "--infile", help="Compressed results", required=True)
    argument = parser.parse_args()
    return argument


def main():
    args = arg_parser()
    compressed_file = args.infile

    # Reading results from file
    with open(compressed_file) as results_file:
        results = json.load(results_file)
    pdfset_name = results["pdfset_name"]
    index = results["index"]  # Array of index
    nbcomp = len(index)

    # Get LHAPDF datadir
    lhapdf_dir = Popen(["lhapdf-config", "--datadir"], stdout=PIPE)
    pdf_dir, _ = lhapdf_dir.communicate()
    pdf_dir = pdf_dir.decode("utf-8")
    pdf_dir = pdf_dir.replace("\n", "")

    # Create output file
    output = pdfset_name + "_compressed_" + str(nbcomp)

    # Create Output folders
    if not os.path.exists(output):
        os.mkdir(output)
    else:
        pass

    src_str = pdf_dir + "/" + pdfset_name + "/" + pdfset_name
    dst_str = output + "/" + output

    # Copy the LHAPDF replicas to the output file
    cindex = []
    console.print(
            "\n• Copying the selected replicas to compressed set:",
            style="bold blue"
    )
    with trange(nbcomp) as iter_index:
        for ix, idx in enumerate(iter_index):
            indx = int(index[idx])  # Make sure it's an integer
            cindex.append(indx)
            # Extension name corresponding to the prior
            # Number of PDF replicas < 10000
            if indx < 10:
                ext_name_prior = "000" + str(indx)
            elif indx < 100:
                ext_name_prior = "00" + str(indx)
            elif indx < 1000:
                ext_name_prior = "0" + str(indx)
            else:
                ext_name_prior = str(indx)
            # Extension name for Compressed replicas
            if (ix + 1) < 10:
                ext_name_compress = "000" + str(ix + 1)
            elif (ix + 1) < 100:
                ext_name_compress = "00" + str(ix + 1)
            elif (ix + 1) < 1000:
                ext_name_compress = "0" + str(ix + 1)
            else:
                ext_name_compress = str(ix + 1)
            # Construc path name
            src = src_str + "_" + ext_name_prior + ".dat"
            dst = dst_str + "_" + ext_name_compress + ".dat"
            # copy srouce to compressed
            shutil.copy(src, dst)
            iter_index.set_description(
                    f"copy original_{indx} to compressed_{ix+1}"
            )

    # Construct .info file for compressed set
    src = src_str + ".info"
    dst = dst_str + ".info"
    src_file = open(src, "r")
    dst_file = open(dst, "w")
    for line in src_file.readlines():
        if "NumMembers:" in line:
            dst_file.write("NumMembers: " + str(nbcomp + 1) + "\n")
        else:
            dst_file.write(line)
    dst_file.close()
    src_file.close()

    # Fetching info from Prior Central PDF
    console.print(
            "\n• Fetching information from original central PDF.",
            style="bold blue"
    )
    w = open(src_str + "_0000.dat", "r")
    xpdf = []
    xgrid, qgrid, fgrid = [], [], []
    textxs, textqs, textfs = [], [], []

    # Removing the info in the head
    for _ in range(0, 10):
        if "--" in w.readline():
            break

    # Init grid size count
    s = 0
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("N", justify="left")
    table.add_column("Subgrids", justify="center", width=35)
    while True:
        textxs.append(w.readline())
        xs = [float(el) for el in textxs[s].split()]
        textqs.append(w.readline())
        qs = [float(el) for el in textqs[s].split()]
        textfs.append(w.readline())
        fs = [int(float(el)) for el in textfs[s].split()]
        if len(xs) == 0:
            break
        xgrid.append(xs)
        qgrid.append(qs)
        fgrid.append(fs)
        nx = len(xgrid[s])
        nq = len(qgrid[s])
        table.add_row(
            f"{s}",
            f"{len(xgrid[s])} {len(qgrid[s])} {len(fgrid[s])}"
        )
        for ix in range(0, nx):
            for iq in range(0, nq):
                w.readline().split()
        w.readline()
        s += 1
    w.close()
    console.print(table)

    # Reading XPDF
    console.print(
            "\n• Extract grid information from compressed set:",
            style="bold blue"
    )
    pdf = lhapdf.mkPDFs(pdfset_name)
    with trange(len(cindex)) as iter_index:
        for irep in iter_index:
            iter_index.set_description(f"Reading Replica {irep}")
            xpdf.append([])
            for ss in range(s):
                xpdf[irep].append([])
                for ix in range(len(xgrid[ss])):
                    xpdf[irep][ss].append([])
                    for iq in range(len(qgrid[ss])):
                        xpdf[irep][ss][ix].append([])
                        for ifl in range(len(fgrid[ss])):
                            xpdf[irep][ss][ix][iq].append(
                                pdf[cindex[irep]].xfxQ(
                                    fgrid[ss][ifl], xgrid[ss][ix], qgrid[ss][iq]
                                )
                            )


    # Construct commpressed central PDF
    console.print("\n• Computing central replicas.", style="bold blue")
    w = open(dst_str + "_0000.dat", "w")
    w.write("PdfType: central\n")
    w.write("Format: lhagrid1\n---\n")

    for ss in range(s):
        w.write(textxs[ss])
        w.write(textqs[ss])
        w.write(textfs[ss])
        for ix in range(len(xgrid[ss])):
            for iq in range(len(qgrid[ss])):
                w.write(" ")
                for ifl in range(len(fgrid[ss])):
                    sum = 0
                    for irep in range(len(cindex)):
                        sum += xpdf[irep][ss][ix][iq][ifl]
                    sum /= nbcomp
                    print("%14.7E" % sum, end=' ', file=w)
                w.write("\n")
        w.write("---\n")
    w.close()


if __name__ == "__main__":
    main()
