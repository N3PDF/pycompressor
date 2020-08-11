#!/usr/bin/python3
import os
import sys
import json
import shutil
import lhapdf
from tqdm import trange
from subprocess import PIPE
from subprocess import Popen


# Check input arguments
# if len(sys.argv) > 1:
#     raise IOError("Only one argument is allowed.")

compressed_file = sys.argv[1]
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
print("\n[+] Copying the selected replicas to compressed set:")
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
        iter_index.set_description(f"copy original_{indx} to compressed_{ix+1}")

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
print("\n[+] Fetching information from original central PDF:")
w = open(src_str + "_0000.dat", "r")
xpdf = []
xgrid, qgrid, fgrid = [], [], []
textxs, textqs, textfs = [], [], []
# Remove the first 3 lines
for j in range(0, 3):
    w.readline()
s = 0
while True:
    w.readline()
    textxs.append(w.readline())
    xs = [float(el) for el in textxs[s].split()]
    textqs.append(w.readline())
    qs = [float(el) for el in textqs[s].split()]
    textfs.append(w.readline())
    fs = [int(el) for el in textfs[s].split()]
    if len(xs) == 0:
        break
    xgrid.append(xs)
    qgrid.append(qs)
    fgrid.append(fs)
    nx = len(xgrid[s])
    nq = len(qgrid[s])
    print(f" - Subgrid {s} {len(xgrid[s])} {len(qgrid[s])} {len(fgrid[s])}")
    for ix in range(0, nx):
        for iq in range(0, nq):
            w.readline().split()
    s += 1
w.close()

# Reading XPDF
print("\n[+] Computing PDF from LHAPDF:")
pdf = lhapdf.mkPDFs(pdfset_name)
print("\n[+] Computing x-pdf:")
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
print("\n[+] Computing central replicas for compressed set.\n")
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
