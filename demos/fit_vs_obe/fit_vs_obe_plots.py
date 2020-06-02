"""
An "experiment" comparing OptBayesExpt class with measure-then-fit results
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import glob
from optbayesexpt import trace_sort

datadir = "fit_vs_obe_data"

if path.isdir(datadir) is not True:
    raise IOError(f'Directory {datadir} not found. Maybe unzip fit_vs_obe_data.zip?')

def lorentz(x, x0, a, b, d):
    """
    Calculate a Lorentzian function of x

    All parameters may be scalars or they may be arrays
        - as long as the arrays interact nicely
    :param x:  measurement setting
    :param a:  Amplitude parameter
    :param b:  background parameter
    :param d:  half-width at half-max parameter
    :param x0: peak center value parameter
    :return:  y  model output (float)
    """
    return b + a / (((x - x0) / d) ** 2 + 1)


#################################################################
# file reading utilities


def readobesigs():

    obefiles = glob.glob(datadir + '/obe???.txt')

    n_trace  = np.loadtxt(obefiles[0], unpack=True)[0]
    
    sigdata = []
    x0data = []
    for file in obefiles:
        try:
            (_, _, _, x0trace, sigtrace) = np.loadtxt(file, unpack=True)
            sigdata.append(sigtrace)
            x0data.append(x0trace)
        except ValueError:
            pass

    return n_trace, x0data, sigdata


def readscansigs():

    scanfiles = glob.glob(datadir + '/scan???.txt')

    n_trace = np.loadtxt(scanfiles[0], unpack=True, )[0]

    sigdata = []
    x0data = []
    for file in scanfiles:
        (_, x0trace, sigtrace) = np.loadtxt(file, unpack=True)
        x0data.append(x0trace)
        sigdata.append(sigtrace)

    return n_trace, x0data, sigdata


def handle_the_truth():
    true_vals = np.loadtxt(path.join(datadir, 'true_values.txt'))
    return true_vals


##################################################################
# Plotting routines
##################################################################


def plot_obe_example(host_axes, obe_filename):

    obe_path = path.join(datadir, obe_filename)
    index, xtrace, ytrace, x0, sigma = np.loadtxt(obe_path, unpack=True)

    axR = host_axes
    axL = axR.twinx()

    xbar, ybar, sbar, nbar = trace_sort(xtrace, ytrace)

    print(np.max(sbar))
    print(np.min(sbar))

    # histogram
    axR.hist(xtrace, bins=len(xbar), zorder=1, range=(1.45, 4.55),
             color='lightblue')
    axR.yaxis.tick_right()
    axR.yaxis.set_label_position('right')
    axR.set_ylabel('measurement count')
    axR.set_xlabel('measurement setting')

    # points & curves
    axL.yaxis.tick_left()
    axL.set_ylabel('measured values')
    axL.yaxis.set_label_position('left')
    axL.set_ylim((-1.0, 2))

    pts = 1 / np.array(sbar)
    plt.scatter(xbar, ybar, s=pts**2, c='blue', alpha=.5,
                label='OptBayesExpt')
    plt.arrow(3, -.45, .7, 0, facecolor="lightblue", edgecolor='None',
              width=.05)
    return axL, axR

def plot_scan_example(host_axes, scan_filename):

    scan_path = path.join(datadir, scan_filename)
    xvals, ymean, ysig = np.loadtxt(scan_path, unpack=True)

    pts = 1 / np.array(ysig)
    host_axes.scatter(xvals, ymean, s=pts**2, c='red', alpha=.5,
                      label='average & fit')


def plot_true_curve(host_axes):
    xlim = host_axes.get_xlim()
    x = np.linspace(*xlim, 200)
    truevals = handle_the_truth()
    plt.plot(x, lorentz(x, *truevals[:-1]), 'k', label='true values')


def plot_obe_traces(host_axes):
    n_obe, x0data, sigdata = readobesigs()

    x0true = trueparams[0]
    rmserr = np.sqrt(np.mean((np.array(x0data) - x0true)**2, axis=0))
    line1, = host_axes.loglog(n_obe, rmserr, "#0000bb", alpha=1)

    # host_axes.loglog(n_obe, sigdata[0], "b", alpha=.05, label="fit sigma")
    for trace in sigdata:
        line2, = host_axes.loglog(n_obe, trace, "b", alpha=.04)
    legend = plt.legend((line1, line2), ("rms mean error", "PDF sigma"),
                        title="OptBayesExpt", loc = 3)
    return legend

def plot_scan_traces(host_axes):
    n_scan, x0data, sigdata = readscansigs()
    x0true = trueparams[0]
    rmserr = np.sqrt(np.mean((np.array(x0data) - x0true)**2, axis=0))
    line3, = host_axes.loglog(n_scan, rmserr, "#bb0000", alpha=1)
    for trace in sigdata:
        line4, = host_axes.loglog(n_scan, trace, "r", alpha=.05)
    legend = plt.legend((line3, line4), ("rms fit error", "fit sigma"),
                        title="average & fit", loc=1)
    return legend

def make_duo_plot():

    global xvals
    plt.rc('font', size=12)
    plt.rc('mathtext', fontset='stixsans')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # first plot: example data
    axL, axR = plot_obe_example(ax1, 'obe000.txt')
    plot_scan_example(axL, 'scanexample.txt')
    plot_true_curve(ax1)
    plt.legend(loc=0)

    # second plot: evolution of uncertainty
    plt.sca(ax2)
    plt.ylim(.005, 1)
    plt.xlim(1, 4e4)
    plt.xlabel("accumulated measurements")
    plt.ylabel("uncertainty of center")

    obe_legend = plot_obe_traces(ax2)
    fit_legend = plot_scan_traces(ax2)    # clobbers the previous legend
    plt.gca().add_artist(obe_legend)
    plt.loglog([1e3, 2e4], [.06, (.06/np.sqrt(20))], "k")
    plt.text(3000, .04, "$N^{-1/2}$")
    plt.tight_layout()
    plt.savefig('rootN.png')
    plt.show()

if __name__ == "__main__":

    truevals = handle_the_truth()
    noiselevel = truevals[-1]
    trueparams = truevals[:-1]

    make_duo_plot()
    plt.show()
