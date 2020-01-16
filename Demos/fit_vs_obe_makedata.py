"""
An "experiment" comparing OptBayesExpt class with measure-then-fit results
"""

import numpy as np
from scipy.optimize import curve_fit
import os.path as path
import os
from multiprocessing import Pool, cpu_count

from optbayesexpt import OptBayesExpt

################################
#  Switches to do/skip different calculations
save_truth = True
example_scans = True
scan_and_fit = True
do_optbayesexpt_runs = True

datadir = "fit_vs_obe_data"

multiprocess = 0
# Uncomment to launch calculations in parallel
multiprocess = cpu_count()

################################
# global variables
# number of settings on the x-axis - common to obe and fit
n_settingvals = 30

# total measurements in a fit run will be n_settingvals * n_fit_scans_per_run
n_fit_scans_per_run = 500

# total measurements in a n obe run
n_obe_iter_per_run = 3000
n_scans = int(n_obe_iter_per_run/n_settingvals)
n_runs = 100

optimum = False
verbose = True
pickiness = 9
noiselevel = 1

####################################################
# set up the OPtBayesExpt instance
####################################################


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

# this is the part where we make use of the BayesOptExpt class
# We inherit from that class and add a model for a particular use
class OptBayesExpt_Lorentz(OptBayesExpt):
    def __init__(self):
        OptBayesExpt.__init__(self)

    def model_function(self, sets, pars, cons):
        # unpack the experimental settings
        x = sets[0]
        # unpack model parameters
        x0 = pars[0]
        a = pars[1]
        b = pars[2]
        # unpack model constants
        d = cons[0]

        return lorentz(x, x0, a, b, d)


# make the instance that we'll use
myOBE = OptBayesExpt_Lorentz()

##############################################################
# describing the universe of our measurement
##############################################################

# the measurement setting space
# values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, n_settingvals)
# incorportate it into myOBE
myOBE.sets = (xvals,)

# the peak's parameter space
# resonance values x0 around 3
x0min = 2
x0max = 4
x0vals = np.linspace(x0min, x0max, 501)
x0resolution = x0vals[1] - x0vals[0]
# peak amplitude
Amin = 0
Amax = 2
Avals = np.linspace(Amin, Amax, 101)
# background
Bmin = -1
Bmax = 1
Bvals = np.linspace(Bmin, Bmax, 51)
# Pack the parameters into a tuple and incorporate it
myOBE.pars = (x0vals,Avals,Bvals)
# note that the parameter order must correspond to how the
# values are unpacked in the model_function

# constant parameters
dtrue = 0.15
# just linewidth here.  Incorportate it.
myOBE.cons = (dtrue,)

# Settings, parameters, constants and model all defined, so set it all up
myOBE.config()
myOBE.Ndraws=20

#################################################################
# Measurement simulation
#################################################################

# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
x0true = 2.5  # pick a random resonance x0
Atrue = 1
Btrue = 0
truevals = (x0true, Atrue, Btrue, dtrue)


def save_true_values(filepath):
    truth = open(filepath, 'w')
    truth.write("#x0\tAtrue\tBtrue\tdtrue\tnoiselevel\n")
    truth.write('{}\t{}\t{}\t{}\t{}\n'.format(x0true, Atrue, Btrue, dtrue, noiselevel))
    truth.close()


def simdata(x):
    # measurement simulator
    global noiselevel
    global truevals
    """
    simulate a measurement at x
    :param x:  measurement setting
    """
    # calculate the theoretical output result
    y = lorentz(x, *truevals)

    # add Gaussian noise
    if type(x) == np.ndarray:
        y += noiselevel * np.random.randn(len(x))
    else:
        y += noiselevel * np.random.randn()
    return y


##################################################################
# optbayesexpt measurement simulations


def oberun(n_measure, n_report=None):

    digits = int(np.log10(n_measure)) + 1
    d_fmt = "N = {:" + "{}".format(digits) + "d}\tx0 = {:5.3f}\tsigma = {:5.3f}"

    if n_report:
        print("{} measurements requested, reporting every {}".format(n_measure, n_report))

    myOBE.set_pdf(flat=True)

    xtrace = []
    ytrace = []
    itrace = []
    x0trace = []
    sigtrace = []

    i = 0
    while i < n_measure:
        """get the measurement seting"""
        if optimum:
            xmeasure, = myOBE.opt_setting()
        else:
            xmeasure, = myOBE.good_setting(pickiness=pickiness)

        """fake measurement"""
        ymeasure = simdata(xmeasure)

        xtrace.append(xmeasure)
        ytrace.append(ymeasure)

        """report the results -- the learning phase"""
        myOBE.pdf_update((xmeasure,), ymeasure, noiselevel)
        # get statistics to track progress
        x0, sig = myOBE.get_mean(0)
        i += 1
        itrace.append(i)
        x0trace.append(x0)
        sigtrace.append(sig)

        if n_report:
            if i % n_report == 0:
                print(d_fmt.format(i, x0, sig))

    return np.array(itrace), np.array(xtrace), np.array(ytrace), np.array(x0trace), np.array(
        sigtrace)


def oberun_to_file(pathname, n_measure=n_obe_iter_per_run):
    # calculate an obe run and save it
    # run the calculation
    index, x, y, x0, sig = oberun(n_measure, n_report=10)
    # shape the results
    databrick = np.array([index, x, y, x0, sig]).T
    # save them
    np.savetxt(pathname, databrick, header="# index, x, y, x0, sig")


##################################################################
# Scanned measurement simulations


def ymodel(x, x0, A, B):
    # a wrapper for the least squares fit to allow fixed peak width (d)
    return lorentz(x, x0, A, B, dtrue)


def simulate_scandata(xvals, scan_count):
    # create a library of simulated scans (for measure then fit method)
    ymeas = np.zeros((scan_count, len(xvals)))
    for i in np.arange(scan_count):
        ymeas[i] = simdata(xvals)
    return ymeas


def multi_scan_stats(ydata):
    # compute averages and standard deviation of a series of scans
    ymean = np.mean(ydata, axis = 0)
    scan_count = ydata.shape[0]

    ysig = np.std(ydata, axis=0)/np.sqrt(scan_count)

    return ymean, ysig


def fitrun(xseries, ydata):
    # fits of increasing accumulation of data

    scan_count = ydata.shape[0]
    n_per_scan = len(xseries)
    n_trail = []
    x0trail = []
    x0sigtrail = []
    for i in np.arange(scan_count):
        ydata_so_far = ydata[:i+1]
        yaverage, ysig = multi_scan_stats(ydata_so_far)
        # initail guesses
        xmax = xseries[np.argmax(yaverage)]
        baseline = np.mean(yaverage)
        amplitude = xmax - baseline
        popt, pcov = curve_fit(ymodel, xseries, yaverage, [xmax, amplitude, baseline])
        x0mean = popt[0]
        x0sigma = np.sqrt(pcov[0, 0])

        n_trail.append((i+1) * n_per_scan)
        x0trail.append(x0mean)
        x0sigtrail.append(x0sigma)

    return n_trail, x0trail, x0sigtrail


def fitrun_to_file(xvalues, n_scans, pathname):

    ydata = simulate_scandata(xvalues, n_scans)
    n_trail, x0mean, x0sig = fitrun(xvalues, ydata)
    databrick = np.array([n_trail, x0mean, x0sig]).T
    np.savetxt(pathname, databrick, header="# n, sigma")


def example_scan_data(xvalues, n_scans, pathname):
    ydata = simulate_scandata(xvalues, n_scans)
    ybar, ysig = multi_scan_stats(ydata)
    databrick = np.array([xvalues, ybar, ysig]).T
    np.savetxt(pathname, databrick)


if __name__ == "__main__":


    if not path.isdir(datadir):
        os.mkdir(datadir)

    #
    # save the true values
    #
    if save_truth:
        truthpath = path.join(datadir, "true_values.txt")
        save_true_values(truthpath)

    if example_scans:
        print("example scans")
        example_scan_data(xvals, n_scans, path.join(datadir, "scanexample.txt"))
    #
    # Calculate scan & fit runs
    #
    # The least-squares routine fails occasionally.
    # the dolist, redolist stuff reruns any failed attempts
    dolist = []
    if scan_and_fit:
        dolist = ['scan{:03d}.txt'.format(i) for i in np.arange(n_runs)]
    redolist = []
    while len(dolist) > 0:
        print(dolist)
        for filename in dolist:
            print(filename)
            scanpath = path.join(datadir, filename)
            try:
                fitrun_to_file(xvals, n_fit_scans_per_run, scanpath)
            except RuntimeError:
                redolist.append(filename)
        dolist = redolist
        redolist = []

    #
    # Calculate obe runs
    #
    if do_optbayesexpt_runs:
        obelist = ['obe{:03d}.txt'.format(i) for i in np.arange(n_runs)]
        obepaths = [path.join(datadir, name) for name in obelist ]

        if multiprocess > 1:
            # run several at a time
            mypool = Pool(processes=multiprocess)
            mypool.map(oberun_to_file, obepaths)
        else:
            # run them one at a time
            for path in obepaths:
                oberun_to_file(path)
