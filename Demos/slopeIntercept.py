"""
Pi pulse tuner

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from OptBayesExpt import OptBayesExpt

"""
ESTABLISH THE EXPERIMENTAL MODEL
"""

# this is the model of our experiment - a straight line
def mxplusb(x, m, b):
    return m*x+b

# this is the part where we make use of the BayesOptExpt class
# We inherit from that class and add a model for a particular use
class OptBayesExpt_slopeIntercept(OptBayesExpt):
    def __init__(self):
        OptBayesExpt.__init__(self)

    def model_function(self, sets, pars, cons):

        # unpack the experimental settings
        x = sets[0]

        # unpack model parameters
        m = pars[0]
        b=pars[1]

        # unpack model constants
        # N/A

        return mxplusb(x, m, b)

# make the instance that we'll use
myOBE = OptBayesExpt_slopeIntercept()

"""
SETTING UP A PARTICULAR EXAMPLE
"""
# define the measurement setting space
# 101 possible x values
xsettings = np.linspace(0, 1, 201)
# sent it to myOBE packaged as a tuple
myOBE.sets = (xsettings,)

# define the parameter space
mvals = np.linspace(-1, 1, 501)
bvals = np.linspace(-1, 1, 501)
# package as a tuple and send
myOBE.pars = (mvals, bvals)

# Settings, parameters, constants and model all defined, so set it all up
myOBE.config()

"""
MEASUREMENT SIMULATION
"""

# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
m_true = 2 * np.random.rand() -1  # pick a random slope betw. -1 and 1
b_true = 2 * np.random.rand() -1  # pick a random intercept

def simdata(x):
    """
    simulate a measurement at pulsetime pt and detuning df
    """
    # calculate the true line
    y = mxplusb(x, m_true, b_true)
    # add noise
    global noiselevel
    s = noiselevel # noise level
    if type(y) == np.ndarray:
        y += s * np.random.randn(len(y))
    else:
        y += s * np.random.randn()
    return y

""" 
RUN THE "EXPERIMENT" DEMO 
"""

def batchplot(subplot):
    global Nmeasure
    global pickiness
    global optimum
    global noiselevel
    global drawplots

    myOBE.set_pdf(flat=True)
    xdata = np.zeros(Nmeasure)
    ydata = np.zeros(Nmeasure)
    mbar = np.zeros(Nmeasure)
    bbar = np.zeros(Nmeasure)
    for i in np.arange(Nmeasure):
        if optimum :
            xset, = myOBE.opt_setting()
        else:
            xset, = myOBE.good_setting(pickiness=pickiness)
        xdata[i] = xset

        measurement = simdata(xset)
        ydata[i] = measurement

        myOBE.pdf_update((xset,), measurement, noiselevel)

    m_mean, sigm = myOBE.get_mean(0)
    b_mean, sigb = myOBE.get_mean(1)

    ytrue = mxplusb(xsettings, m_true, b_true)

    axL = subplot.twinx()
    subplot.hist(xdata, bins=20, color='lightblue')
    subplot.yaxis.tick_right()
    subplot.yaxis.set_label_position('right')
    subplot.set_ylabel("points")
    subplot.set_xlabel('x')
    ymax = Nmeasure/2 * 1.1
    subplot.set_ylim(0, ymax)


    axL.plot(xsettings, ytrue, 'r-')
    axL.errorbar(xdata, ydata, noiselevel, fmt='.', color='k')
    axL.yaxis.tick_left()
    axL.yaxis.set_label_position('left')

    axL.set_ylabel('y')
    bottom, top = axL.get_ylim()
    bottom -= .5
    top += .3
    axL.set_ylim((bottom, top))
    y1 = bottom + .93 * (top - bottom)
    y2 = bottom + .85 * (top - bottom)
    plt.text(0.05, y1 , 'm = {:5.3f}$\pm${:5.3f}'.format(m_mean,sigm))
    plt.text(0.05, y2 , 'b = {:5.3f}$\pm${:5.3f}'.format(b_mean,sigb))

    return subplot.yaxis, axL.yaxis

Nmeasure = 40
noiselevel = .1

fig, axes = plt.subplots(1, 4, figsize=(12, 3))


optimum = True
batchplot(axes[0])
plt.title('opt_setting()')


optimum=False
pickiness = 9
batchplot(axes[1])
plt.title('good_setting(pickiness={})'.format(pickiness))

optimum=False
pickiness = 4
batchplot(axes[2])
plt.title('good_setting(pickiness={})'.format(pickiness))

optimum=False
pickiness = 1
batchplot(axes[3])
plt.title('good_setting(pickiness={})'.format(pickiness))

plt.tight_layout()
plt.show()

# import cProfile
# cProfile.run('batchdemo()', sort='cumtime')
