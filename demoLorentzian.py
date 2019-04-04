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
        #d = cons[0]
        d = pars[3]

        return Lorentz(x, x0, A, B, d)

# make the instance that we'll use
myOBE = OptBayesExpt_Lorentz()

"""
SETTING UP A PARTICULAR EXAMPLE
"""
# define the measurement setting space
# 50 values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, 200)
# tell it to the BOE
# sets, pars, cons are all expected to be tuples
myOBE.sets = (xvals,)

# define the parameter space where the peak could be found
# resonance values x0 (like NV frequency) around 3 GHz
x0min = 2
x0max = 4
x0vals = np.linspace(x0min, x0max, 151)
x0resolution = x0vals[1]-x0vals[0]
# peak amplitude
Amin = 2
Amax = 5
Avals = np.linspace(Amin, Amax, 101)
# background
Bmin = -1
Bmax = 1
Bvals = np.linspace(Bmin, Bmax, 51)

dmin = .05
dmax = .2
dvals = np.linspace(dmin, dmax, 15)

# Pack the parameters into a tuple and send it to the BOE
# note that the order must correspond to how the values are unpacked in the model_function
myOBE.pars = (x0vals,Avals,Bvals,dvals)

# define the known constants
# keeping the peak width constant in this example
# dtrue = .1
# myOBE.cons = (dtrue,)

# Settings, parameters, constants and model all defined, so set it all up
myOBE.config()

"""
MEASUREMENT SIMULATION
"""

# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
x0true = (x0max - x0min) * np.random.rand() + x0min  # pick a random resonance x0
Btrue = (Bmax - Bmin) * np.random.rand() + Bmin  # pick a random background
Atrue = (Amax - Amin) * np.random.rand() + Amin  # pick a random amplitude
dtrue = (dmax-dmin) * np.random.rand() + dmin # pick a random width

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

def batchdemo():
    global Nmeasure
    global pickiness
    global optimum
    global noiselevel
    global drawplots

# first the measure-then-fit method
    xseries = np.linspace(xvals[0], xvals[-1], Nmeasure)
    yseries = simdata(xseries)
    xmax = xseries[np.argmax(yseries)]

    popt, pcov = curve_fit(ymodel, xseries, yseries, [xmax, 1, 0, .1])
    print(popt, pcov)

    x0fit, Afit, Bfit, dfit = popt
    x0sigma = np.sqrt(pcov[0,0])
    plt.figure(figsize=(9,4))
    ytop = Amax+noiselevel
    ylims = (Bmin-2*noiselevel, ytop)

    plt.subplot(121)

    plt.errorbar(xseries, yseries, yerr=noiselevel, fmt='.', color='k')
    xsmooth = np.linspace(xvals[0], xvals[-1], 400)

    plt.plot(xsmooth, ymodel(xsmooth, *popt), '#66EE66', linewidth=3, label='best fit')
    plt.plot(xsmooth,ymodel(xsmooth,x0true, Atrue, Btrue, dtrue), color='red', label="true curve")
    plt.legend(loc=1)
    plt.ylim(ylims)
    plt.xlim((1.5, 4.5))
    plt.text(1.6, .93*ytop, '{} measurements'.format(Nmeasure))
    plt.text(1.6, .85*ytop, 'fit: x0 = {:6.4f} $\pm$ {:6.4f}'.format(x0fit, x0sigma))
    plt.text(1.6, .80*ytop, 'true x0 = {:6.4f}'.format(x0true))
    plt.xlabel('x setting')
    plt.ylabel('y result')

# next the OptBayesExpt

    xdata = np.zeros(Nmeasure)
    ydata = np.zeros(Nmeasure)
    sig = np.zeros(Nmeasure)
    for i in np.arange(Nmeasure):
        """get the measurement seting"""
        if optimum:
            xmeasure, = myOBE.opt_setting()
        else:
            xmeasure, = myOBE.good_setting(pickiness=pickiness)

        """fake measurement"""
        ymeasure = simdata(xmeasure)
        xdata[i] = xmeasure
        ydata[i] = ymeasure

        """report the results -- the learning phase"""
        myOBE.pdf_update((xmeasure,), ymeasure, noiselevel)
        # get statistics to track progress
        x0, sig[i] = myOBE.get_mean(0)

        print(i, "x0 = {}, sigma = {}".format(x0, sig[i]))
        if sig[i] < x0resolution: break
        if sig[i] < x0sigma: break

    plt.subplot(122)

    # Plot different model curves based on random draws from the parameter distribution
    paramsets = myOBE.markov_draws()
    # the default number of draws is myOBE.Ndraws, set in ProbDistFunc__init__()

    # fill the model results for each drawn parameter set
    ycalc = myOBE.eval_over_all_settings(paramsets[0])
    plt.plot(xvals, ycalc, color='g', alpha=.05, linewidth=3, label='parameter draw')

    for oneparamset in paramsets[1:drawplots] :
        ycalc = myOBE.eval_over_all_settings(oneparamset)
        plt.plot(xvals, ycalc, color='g', alpha=.05, linewidth=3, )
    # plot the 'data'
    plt.errorbar(xdata, ydata, yerr=noiselevel, fmt='.', color='k')
    # true curve
    plt.plot(xvals, Lorentz(xvals, x0true, Atrue, Btrue, dtrue), color='red', label='true curve')
    plt.legend(loc=1)

    plt.ylim(ylims)
    plt.xlim((1.5, 4.5))
    plt.text(1.6, .93*ytop, '{} measurements'.format(i+1))
    plt.text(1.6, .85*ytop, 'fit: x0 = {:6.4f} $\pm$ {:6.4f}'.format(x0, sig[i]))
    plt.text(1.6, .80*ytop, 'true x0 = {:6.4f}'.format(x0true))
    plt.xlabel('x setting')

    plt.tight_layout()
    plt.show()


Nmeasure = 30

smartmeasure = True
optimum = True
pickiness = 7
noiselevel = 1
drawplots = 50

batchdemo()
