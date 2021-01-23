#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


#################################################
# Deine list of Estimators for the C compressor #
# results formatting                            #
#################################################

LIST_STAT = [
        "rcv",
        "rmd",
        "l50",
        "u50",
        "l68",
        "u68",
        "l90",
        "u90"
]

LIST_ESTM = [
        "mean",
        "stdev",
        "skewness",
        "kurtosis",
        "kilomogorov_smirnov",
        "correlation"
]


def axes_width(ax, lw=1):
    """Change width of axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Figure's axes
    lw : float
        Define width of axes
    """

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)


def compute_errobars(lower, upper):
    """Given a lower and upper value, compute the
    central value and the error.

    Parameters
    ----------
    lower : float
        Lower value
    upper : float
        Upper value
    """

    low = np.array(lower)
    upp = np.array(upper)
    central = (upp + low) / 2
    error = upp - central
    return central, error


#####################################################################
# Routines that read and extract the metadata from the C Compressor #
#####################################################################


def extract_data(infile):
    """Read and extract the metadata from the C compressor
    output.

    Parameters
    ----------
    infile : file
        File containing the results of the compressor
    """

    results = []
    readfile = open(infile, "r").readlines()
    for line in readfile:
        newline = line.strip("\n").split("\t")
        arr_val = [float(val) for val in newline[:-1]]
        results.append(arr_val)
    return results


def parse_cccomp_random(infile):
    """Extract the results from random trials.

    Parameters
    ----------
    infile : file
        File containing the results of the compressor
    """

    new_dic = {}
    results = extract_data(infile)
    index_comp = [row[0] for row in results]
    for position, est in enumerate(LIST_ESTM):
        position *= 8
        stat_dic = {}
        for indx, stat in enumerate(LIST_STAT, start=1):
            arr_vals = []
            for res in results:
                arr_vals.append(res[position + indx])
                stat_dic[stat] = arr_vals
            new_dic[est] = stat_dic
    return index_comp, new_dic


def parse_cccomp_reduced(infile):
    """Extract the final values of ERF.

    Parameters
    ----------
    infile : file
        File containing the results of the compressor
    """

    new_dic = {}
    results = extract_data(infile)
    index_comp = [row[0] for row in results]
    for position, est in enumerate(LIST_ESTM, start=1):
        arr_vals = []
        for res in results:
            arr_vals.append(res[position])
        new_dic[est] = arr_vals
    return index_comp, new_dic


#####################################################################
# Routines that read and extract the metadata from the pyCompressor #
#####################################################################


def extract_dic(infile):
    """Read and extract the metadata from the pyCompressor
    output.

    Parameters
    ----------
    infile : file
        File containing the results of the compressor
    """

    results = []
    readfile = open(infile, "r").readlines()
    for line in readfile:
        infoline = line.strip("\n").split(":", 1)
        rep_size = int(infoline[0])
        dic_estm = eval(infoline[1])
        results.append([rep_size, dic_estm])
    return results


def parse_pycomp_random(infile):
    """Extract the results from random trials.

    Parameters
    ----------
    infile : file
        File containing the results of the compressor
    """

    new_dic = {}
    results = extract_dic(infile)
    list_cmp = [row[0] for row in results]
    list_est = list(results[0][1].keys())
    list_stt = list(results[0][1][list_est[0]].keys())
    for est in list_est:
        val_dic = {}
        for stat in list_stt:
            arr_vals = []
            for val in results:
                arr_vals.append(val[1][est][stat])
            val_dic[stat] = arr_vals
        new_dic[est] = val_dic
    return list_cmp, new_dic


def parse_pcycomp_reduced(infile):
    """Extract the final values of ERF.

    Parameters
    ----------
    infile : file
        File containing the results of the compressor
    """

    new_dic = {}
    results = extract_dic(infile)
    list_cmp = [row[0] for row in results]
    list_est = list(results[0][1].keys())
    for est in list_est:
        arr_vals = []
        for val in results:
            arr_vals.append(val[1][est])
        new_dic[est] = arr_vals
    return list_cmp, new_dic


def plot_erfs(erf_random, erf_reduced, formatting="pycomp"):
    """plot_erfs.

    Parameters
    ----------
    erf_random :
        erf_random
    erf_reduced :
        erf_reduced
    formatting :
        formatting
    """

    if formatting == "pycomp":
        indx, erf_vals = parse_pycomp_random(erf_random)
        indx, erf_tots = parse_pcycomp_reduced(erf_reduced)
    elif formatting == "ccomp":
        indx, erf_vals = parse_cccomp_random(erf_random)
        indx, erf_tots = parse_cccomp_reduced(erf_reduced)
    else:
        raise ValueError("Format not recognised.")

    # Construct Grid Plots
    list_est = list(erf_vals.keys())
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=[20.8, 20.8])

    # Loop over the grid
    for i, axis in enumerate(axes.reshape(-1)):
        estmd_stat = erf_vals[list_est[i]]
        estmd_tots = erf_tots[list_est[i]]
        axis.set_title(list_est[i], fontsize=16)
        axis.set_yscale('log')
        # c.i.
        c50, error50 = compute_errobars(estmd_stat["l50"], estmd_stat["u50"])
        c68, error68 = compute_errobars(estmd_stat["l68"], estmd_stat["u68"])
        c90, error90 = compute_errobars(estmd_stat["l90"], estmd_stat["u90"])
        # Actual Plotting
        axis.errorbar(
                indx,
                c90,
                yerr=error90,
                linestyle="",
                ecolor="#2ca02c",
                elinewidth=2.25,
                capsize=6,
                zorder=1,
                label="Raondom 50% c.l. (1k)"
        )
        axis.errorbar(
                indx,
                c68,
                yerr=error68,
                linestyle="",
                ecolor="#ff7f0e",
                elinewidth=2.25,
                capsize=6,
                zorder=2,
                label="Raondom 68% c.l. (1k)"
        )
        axis.errorbar(
                indx,
                c50,
                yerr=error50,
                linestyle="",
                ecolor="#1f77b4",
                elinewidth=2.25,
                capsize=6,
                zorder=3,
                label="Raondom 90% c.l. (1k)"
        )
        axis.scatter(
                indx,
                estmd_stat["rcv"],
                s=80,
                color="#ff00ff",
                zorder=4,
                label="Random Mean (1k)"
        )
        axis.scatter(
                indx,
                estmd_stat["rmd"],
                s=80,
                color="#0000ff",
                zorder=4,
                label="Random Median (1k)"
        )
        # Total ERFs
        axis.scatter(
                indx,
                estmd_tots,
                s=80,
                color="#d62728",
                zorder=5,
                label="compressed"
        )
        # Parameters
        axes_width(axis, lw=1.5)
        axis.grid(alpha=0.1, linewidth=1.5)
        axis.tick_params(length=7, width=1.5)
        axis.tick_params(which='minor', length=4, width=1)
        handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=16)
    fig.savefig("pycompressor_validation.png", dpi=350)


def arg_parser():
    """Parse inputs data file"""

    parser = argparse.ArgumentParser(description="Plot ERFs validation plot.")
    parser.add_argument("--random", help="Random trials results", required=True)
    parser.add_argument("--reduced", help="Final ERFs values", required=True)
    parser.add_argument("--format", help="Metadata format", default="pycomp")
    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    args = arg_parser()
    random = args.random
    reduced = args.reduced
    formatting = args.format

    plot_erfs(random, reduced, formatting=formatting)
