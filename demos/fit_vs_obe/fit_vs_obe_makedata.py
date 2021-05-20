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

multiprocess = True

################################
# global variables
n_settingvals = 30

n_fit_scans_per_run = 500
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

def my_model_function(sets, pars, cons):
    # unpack the experimental settings
    x = sets[0]
    # unpack model parameters
    x0 = pars[0]
    a = pars[1]
    b = pars[2]
    # unpack model constants
    d = cons[0]
    return lorentz(x, x0, a, b, d)

##############################################################
# describing the universe of our measurement
##############################################################

# the measurement setting space
# 200 values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, n_settingvals)
# incorportate it into myOBE
sets = (xvals,)

n_particles = 100000
# the peak's parameter space
# resonance values x0 around 3
x0min = 2
x0max = 4
x0_draws = np.random.uniform(x0min, x0max, n_particles)
# peak amplitude
a_scale = 1
a_draws = np.random.exponential(a_scale, n_particles)
# background
b_min = -1
b_max = 1
b_draws = np.random.uniform(b_min, b_max, n_particles)
# Pack the parameters into a tuple and incorporate it
pars = (x0_draws, a_draws, b_draws)
# note that the parameter order must correspond to how the
# values are unpacked in the model_function

# constant parameters -- just linewidth here.
dtrue = 0.15
cons = (dtrue,)

# Settings, parameters, constants and model all defined, so set it all up
myOBE = OptBayesExpt(my_model_function, sets, pars, cons,
                     n_draws=30, scale=False)

initial_pars = myOBE.parameters

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

    myOBE.set_pdf(initial_pars)

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
        myOBE.pdf_update(((xmeasure,), ymeasure, noiselevel))
        # get statistics to track progress
        x0 = myOBE.mean()[0]
        sig = myOBE.std()[0]
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
    print(pathname)
    index, x, y, x0, sig = oberun(n_measure, n_report=100)
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


def fitrun_to_file(xvalues, scans, pathname):

    ydata = simulate_scandata(xvalues, scans)
    n_trail, x0mean, x0sig = fitrun(xvalues, ydata)
    databrick = np.array([n_trail, x0mean, x0sig]).T
    np.savetxt(pathname, databrick, header="# n, sigma")


def example_scan_data(xvalues, scans, pathname):
    ydata = simulate_scandata(xvalues, scans)
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
    # Calculate obe runs, in parallel if possible
    #
    obelist = []
    if do_optbayesexpt_runs:
        obelist = ['obe{:03d}.txt'.format(i) for i in np.arange(n_runs)]
        obepaths = [path.join(datadir, name) for name in obelist ]
        if multiprocess:
            mypool = Pool(cpu_count()-1)
            mypool.map(oberun_to_file, obepaths)
        else:
            for path in obepaths:
                oberun_to_file(path)


