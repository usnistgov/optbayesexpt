"""
A simple comparing OptBayesExpt class with measure-then-fit.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from OptBayesExpt import OptBayesExpt

"""
ESTABLISH THE EXPERIMENTAL MODEL
"""

# this is the model of our experiment - a Lorentzian peak.
def Lorentz(x, x0, A, B, d):
    """
    Calculate a Lorentzian function of x
    All parameters may be scalars or they may be arrays
        - as long as the arrays interact nicely
    :param x:  measurement setting
    :param A:  Amplitude parameter
    :param B:  background parameter
    :param d:  half-width at half-max parameter
    :param x0: peak center value parameter
    :return:  y  model output (float)
    """
    return B + A / (((x - x0) / d) ** 2 + 1)


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
        A = pars[1]
        B = pars[2]
        # unpack model constants
        d = cons[0]

        return Lorentz(x, x0, A, B, d)

# make the instance that we'll use
myOBE = OptBayesExpt_Lorentz()

"""
SETTING UP A PARTICULAR EXAMPLE
"""
# define the measurement setting space
# 200 values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, 20)
# tell it to the BOE
# sets, pars, cons are all expected to be tuples
myOBE.sets = (xvals,)

# define the parameter space where the peak could be found
# resonance values x0 around 3
x0min = 2
x0max = 4
x0vals = np.linspace(x0min, x0max, 501)
x0resolution = x0vals[1]-x0vals[0]
# peak amplitude
Amin = 2
Amax = 5
Avals = np.linspace(Amin, Amax, 101)
# background
Bmin = -1
Bmax = 1
Bvals = np.linspace(Bmin, Bmax, 51)

# Pack the parameters into a tuple and send it to the BOE
# note that the order must correspond to how the values are unpacked in the model_function
myOBE.pars = (x0vals,Avals,Bvals)

# constant linewidth
dtrue = 0.3
myOBE.cons = (dtrue,)

# Settings, parameters, constants and model all defined, so set it all up
myOBE.config()
myOBE.Ndraws=20

"""
MEASUREMENT SIMULATION
"""

# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
x0true = (2.8-2.2) * np.random.rand() + 2.2  # pick a random resonance x0
# Btrue = (Bmax - Bmin) * np.random.rand() + Bmin  # pick a random background
# Atrue = (Amax - Amin) * np.random.rand() + Amin  # pick a random amplitude
Atrue = 1
Btrue = 0


def simdata(x):
    """
    simulate a measurement at x
    :param x:  measurement setting
    """
    # calculate the theoretical output result
    y = Lorentz(x, x0true, Atrue, Btrue, dtrue)
    # add noise
    global noiselevel
    s = noiselevel

    if type(x) == np.ndarray:
        y += s * np.random.randn(len(x))
    else:
        y += s * np.random.randn()
    return y

""" 
RUN THE "EXPERIMENT" DEMO 
"""

def ymodel(x, x0, A, B, d):
    return Lorentz(x, x0, A, B, d)

def fititerate():

    xseries = np.linspace(xvals[0], xvals[-1], Nmeasure)
    ysum = np.zeros(xseries.shape)
    ysquare = np.zeros(xseries.shape)
    Ntrail = []
    x0sigtrail = []
    for i in np.arange(1,Nbatch+1):
        print(i)
        yvals = simdata(xseries)
        ysum += yvals
        ysquare += yvals**2
        yaverage = ysum/i
        xmax = xseries[np.argmax(ysum)]

        try:
            popt, pcov = curve_fit(ymodel, xseries, yaverage, [xmax, 1, 0, .1])
            x0fit, Afit, Bfit, dfit = popt
            x0sigma = np.sqrt(pcov[0, 0])
            Ntrail.append(i * Nmeasure)
            x0sigtrail.append(x0sigma)
        except RuntimeError:
            print('hiccup')


        if i == Nscans:
            sig = np.sqrt(ysquare/i - yaverage**2)/np.sqrt(i*1.0)
            trace = open('trace10.txt', 'w')
            for x, y, s in zip(xseries, yaverage, sig):
                trace.write('{}\t{}\t{}\n'.format(x, y, s))
            trace.close()

    return(Ntrail, x0sigtrail)


def batchdata():
    global Nmeasure
    global pickiness
    global optimum
    global noiselevel
    global drawplots

    truth = open('truth.txt', 'w')
    truth.write('{}\t{}\t{}\t{}\n'.format(x0true, Atrue, Btrue, dtrue))
    truth.close()

    N, x0 = fititerate()
    fitfile = open('fitdata.txt', "w")
    for anN, anx0 in zip(N, x0):
        fitfile.write('{}\t{}\n'.format(anN, anx0))
    fitfile.close()
    truth = open('truth.txt', 'w')
    truth.write('{}\t{}\t{}\t{}\n'.format(x0true, Atrue, Btrue, dtrue))
    truth.close()

    sig = 3*x0resolution
    xtrace = []
    ytrace = []
    i = 0
    obefile = open('obedata.txt', "w")

    while sig > .01:
        """get the measurement seting"""
        if optimum:
            xmeasure, = myOBE.opt_setting()
        else:
            xmeasure, = myOBE.good_setting(pickiness=pickiness)

        """fake measurement"""
        ymeasure = simdata(xmeasure)

        if i < Nmeasure*Nscans:
            xtrace.append(xmeasure)
            ytrace.append(ymeasure)

        """report the results -- the learning phase"""
        myOBE.pdf_update((xmeasure,), ymeasure, noiselevel)
        # get statistics to track progress
        x0, sig = myOBE.get_mean(0)
        i += 1
        obefile.write('{}\t{}\t{}\n'.format(i, x0, sig))
        print(i, "x0 = {:6.4f}, sigma = {:6.4f}".format(x0, sig))

    obefile.close()
    obetrace = open('obetrace.txt', 'w')
    for x, y in zip(xtrace, ytrace):
        obetrace.write('{}\t{}\n'.format(x, y))
    obetrace.close()


def batchplot():

    # plt.rc('font', size=12)
    # plt.rc('mathtext', fontset='stixsans')

    scantrace = np.loadtxt('trace10.txt', unpack=True)
    obetrace = np.loadtxt('obetrace.txt', unpack=True)

    obex = obetrace[0]
    obey = obetrace[1]
    sortindices = np.argsort(obex)
    sobex = obex[sortindices]
    sobey = obey[sortindices]

    oldx = sobex[0]
    xbar = []
    ybar = []
    sbar = []
    ylist = []
    for x, y in zip(sobex, sobey):
        if x != oldx:
            # new x value
            # process the accumulated data
            xbar.append(oldx)
            ybar.append(np.mean(np.array(ylist)))
            sbar.append(noiselevel/np.sqrt(len(ylist)))
            # reset accumulation
            oldx = x
            ylist = [y, ]
        else:
            # accumulate
            ylist.append(y)
    # oh, and the last batch
    xbar.append(oldx)
    ybar.append(np.mean(np.array(ylist)))
    sbar.append(noiselevel/np.sqrt(len(ylist)))

    # PLOTS
    plt.figure(figsize=(10, 4))
    axR = plt.subplot(121)
    axL = axR.twinx()

    # histogram

    axR.hist(obex, bins=20, zorder=1, range=(1.45, 4.55), color='lightblue')
    axR.yaxis.tick_right()
    axR.yaxis.set_label_position('right')
    axR.set_ylabel('measurement count')
    axR.set_xlabel('measurement setting')

    # points & curves

    axL.yaxis.tick_left()
    axL.set_ylabel('measured values')
    axL.yaxis.set_label_position('left')
    axL.set_ylim((-1.0, 2))


    # plt.plot(obetrace[0], obetrace[1], 'o')
    # OBE values
    # plt.errorbar(xbar, ybar, sbar, fmt = 'o', lw=3, color = 'blue',
    #              label='OptBayesExpt', zorder=5)
    smax = np.max(sbar)
    pts = smax / sbar
    plt.scatter(xbar, ybar, s=pts**2, c='blue', label='OptBayesExpt')

    #scan values
    sig = noiselevel/np.sqrt(Nscans)
    # axL.errorbar(scantrace[0], scantrace[1], sig, fmt='.', lw=1, capsize=3,  color='red',
    #              label='average & fit')
    plt.scatter(scantrace[0], scantrace[1], s= (smax/sig)**2, c='red', label='average & fit')

    truevals = np.loadtxt('truth.txt')
    x =np.arange(1.5, 4.5, .01)
    plt.plot(x, Lorentz(x, *truevals), 'k', label='true values')
    plt.legend(loc=1)
    plt.arrow(3, -.45, .7, 0, facecolor="lightblue", edgecolor='None', width=.05)

    # Log-Log
    plt.subplot(122)

    obedata = np.loadtxt('obedata.txt', unpack=True)
    obe_i = obedata[0]
    obe_sig = obedata[2]
    plt.loglog(obe_i, obe_sig, '.', label='OptBayesExpt', color='blue')

    fitdata = np.loadtxt('fitdata.txt', unpack=True)
    fit_i = fitdata[0]
    fit_sig = fitdata[1]
    plt.loglog(fit_i, fit_sig, '.', label='average & fit', color='red')

    plt.legend()
    plt.xlim(left=3)
    plt.ylim(top=.5)
    plt.xlabel('Total measurements')
    plt.ylabel('peak center uncertainty')

    plt.arrow(500, .015, 4500, 0, facecolor='k', edgecolor='None', zorder = 30)
    plt.text(1000, .017, "x 10")


    plt.tight_layout()
    plt.show()


Nmeasure = 20
Nscans = 50
Nbatch = 1000

optimum = False
pickiness = 10
noiselevel = 1

# data calculations stored in .txt files
# batchdata()

# read .txt files and make plots
batchplot()