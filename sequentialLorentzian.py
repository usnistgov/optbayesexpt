"""
A simple example of how to use the BayesOptExpt class to speed up measurements.

This tool addresses the situation where we make measurements in order to determine the
parameters of model.   This is the situation where we would traditionally make a series of
measurements an then do fitting to extract the parameters.

A weakness of the measure-then-fit method is that the information accumulated during measurement
is not revealed until the fitting stage.  You don't learn much until the measurements are all
done.  Also, with measurements that are all determined beforehand, the measure-then-fit method
can waste time on unimportant measurements while

In contrast, the BayesOptExpt class "learns" from each measurement result and then uses that
knowledge to suggest settings for the next measurement.  The "knowledge" is contained in a
probability distribution of parameter values.  The narrower the distribution, the better we
know the parameters. The "learning" process uses Bayes theorem to refine the probability
distribution based on each new measurement.

The benefits of this on-the-fly learning process are reaped when we use the accumulated
knowledge to guide the measurement process.  The learning allows us to make (partially) informed
decisions about measurement settings that are likely to be most useful.  What we mean by "useful"
here needs to be defined in the code, but the process can be sketched as follows.

What we (claim) we know about the world is that measurements will behave like our model, but we're
fuzzy on the model parameters, as described by their probability density.  Since the measurements
and the parameters are connected, if we allow the parameters to vary, the model will predict
correspondingly varying measurement values for fixed measurement settings.  Several lines of
information theory and Bayes theorem yield a very intuitive result: that the best measurement to
make next is the one that will pin down the model where it is varying the most.


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from OptBayesExpt import OptBayesExpt

"""
ESTABLISH THE EXPERIMENTAL MODEL
"""



# this is the model of our experiment - a Lorentzian peak.
# Sometimes people are concerned about Bayesian statistics because the prior can bias the output.
# However, the more serious bias (in my opinion) is in the model.  With the experimental model, we are asserting that
# we know how the world behaves, except for a few parameters that need fitting.
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
# 50 values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, 200)
# tell it to the BOE
# sets, pars, cons are all expected to be tuples
myOBE.sets = (xvals,)

# define the parameter space where the peak could be found
# resonance values x0 (like NV frequency) around 3 GHz
x0min = 2
x0max = 4
x0vals = np.linspace(x0min, x0max, 201)
# peak amplitude
Amin = -2000
Amax = -40000
Avals = np.linspace(Amin, Amax, 101)
# background
Bmin = 275000
Bmax = 225000
Bvals = np.linspace(Bmin, Bmax, 151)

# Pack the parameters into a tuple and send it to the BOE
# note that the order must correspond to how the values are unpacked in the model_function
myOBE.pars = (x0vals, Avals, Bvals)

# define the known constants
# keeping the peak width constant in this example
dtrue = .1
myOBE.cons = (dtrue,)

# Settings, parameters, constants and model all defined, so set it all up
myOBE.config()

"""
MEASUREMENT SIMULATION
"""

# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
x0true = (x0max - x0min) * np.random.rand() + x0min  # pick a random resonance x0
Btrue = (Bmax - Bmin) * np.random.rand() + Bmin  # pick a random background
# Btrue = 250000

Atrue = (Amax - Amin) * np.random.rand() + Amin  # pick a random amplitude

def simdata(x):
    """
    simulate a measurement at x
    :param x:  measurement setting
    """
    # calculate the theoretical output result
    y = Lorentz(x, x0true, Atrue, Btrue, dtrue)
    # add 2% noise
    s = 0.02 * y

    if type(x) == np.ndarray:
        y += s * np.random.randn(len(x))
    else:
        y += s * np.random.randn()
    return y


""" 
RUN THE "EXPERIMENT" DEMO 
"""


def batchdemo():
    global Nmeasure
    global pickiness
    global optimum

    xdata = np.zeros(Nmeasure)
    ydata = np.zeros(Nmeasure)
    sig = np.zeros(Nmeasure)
    for i in np.arange(Nmeasure):
        """ get the optimum measurement seting """
        if optimum:
            xmeas, = myOBE.opt_setting()
        else:
            xmeas, = myOBE.good_setting(pickiness=pickiness)
        xmeasure = reply[0]
        ymeasure = simdata(xmeasure)
        xdata[i] = xmeasure
        ydata[i] = ymeasure

        """report the results -- the learning phase"""
        myOBE.pdf_update((xmeasure,), ymeasure, 0.02 * ymeasure)
        # get statistics to track progress
        sig[i] = myOBE.get_std(0)

        print(i, "sigma = {}".format(sig[i]))

    plt.figure(figsize=(5,8))
    plt.subplot(311)
    plt.plot(xdata, ydata, '.')
    plt.plot(xvals, Lorentz(xvals, x0true, Atrue, Btrue, dtrue))

    plt.subplot(312)
    plt.hist(xdata, bins=20)
    plt.ylabel("x density")

    plt.subplot(313)
    plt.semilogy(sig)
    print(xvals[1]-xvals[0])
    plt.ylabel("sigma")
    plt.tight_layout()
    plt.show()



fig, ax1 = plt.subplots(ncols=1, nrows=1, sharex=True)


def myinit():
    global xvals, fig, ax1

    # initial pass of the plot animation, i.e. the background
    ax1.set_xlim(xvals.min(), xvals.max())
    ax1.set_ylim(200000, 275000)
    ax1.set_xlabel("xvalue")
    ax1.set_ylabel("photon count")
    line1, = ax1.plot([], [], 'k.', markersize=5.0)
    line2, = ax1.plot(xvals, Lorentz(xvals, x0true, Atrue, Btrue, dtrue), 'r-', linewidth=2)

    ax1.grid()

    global xdata, ydata
    xdata = []
    ydata = []

    global lines
    lines = (line1,)
    return lines


def myfunc(data):
    # a function called by the animation to plot updated stuff
    xmeas, ymeas = data

    xdata.append(xmeas)
    ydata.append(ymeas)

    # update measured data plot
    lines[0].set_data(xdata, ydata)

    return lines


def myframes():
    # a generator that comes up with new data for the next frame
    cnt = 0
    global xvals
    global smartmeasure
    global pickiness
    global optimum

    i = 0
    while cnt < Nmeasure:
        cnt += 1
        print(cnt)
        if smartmeasure:
            """ Get the new measurement setting by Bayes Optimization  """
            if optimum:
                xmeas, = myOBE.opt_setting()
            else:
                xmeas, = myOBE.good_setting(pickiness=pickiness)

            ymeas = simdata(xmeas)
            """ report the measurement back in order to update """
            myOBE.pdf_update((xmeas,), ymeas, 0.02 * ymeas)
        else:
            # or we're just going to sweep the x measurement values
            try:
                xmeas = stupidx[i]
            except UnboundLocalError:
                stupidx = np.linspace(xvals[0], xvals[-1], Nmeasure)
                xmeas = stupidx[i]
            ymeas = simdata(xmeas)
            i = (i + 1) % len(xvals)

        yield xmeas, ymeas


def livedemo():
    global fig
    global xdata
    global ax2
    plt.rc('font', size=16)
    ani = animation.FuncAnimation(fig, myfunc, frames=myframes, init_func=myinit, blit=True, interval=0, repeat=False)
    plt.show()
    plt.hist(xdata, bins=40)
    plt.show()

Nmeasure = 100
smartmeasure = True
optimum = False
pickiness = 2

livedemo()
# batchdemo()
