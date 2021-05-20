"""
This script demonstrates the use of OptBayesExpt in a simulated measurement.
The script has three main stages:

1. A setup stage where all the necessary information about the experiment is
   prepared and then used to create an OptBayesExpt object
2. A measurement loop where the OptBayesExpt object selects measurement
   settings, interprets measurement data, and provides statistics of the
   parameter distribution to monitor progress.
3. Finally, a plotting stage to display the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from optbayesexpt import MeasurementSimulator

from obe_sweeper import OptBayesExptSweeper

# Script tuning parameters
#
# the parameter probability distribution is represented by samples from the
# distribution.
n_samples = 50000
# quit measuring after n_measure individual data points
n_measure = 1000
# Setting selection method
optimal = True        # use OptBayesExpt.opt_setting()
# optimal = False         # use OptBayesExpt.good_setting()
pickiness = 20         # ignored when optimal == True
# measurement simulation added noise level
noise_level = 2000
# random number generator
try:
    rng = np.random.default_rng()
except AttributeError:
    rng = np.random

########################################################################
#           OptBayesExpt SETUP
########################################################################

def my_model_function(sets, pars, cons):
    """ Evaluates a trusted model of the experiment's output

    The equivalent of a fit function. The argument structure is
    required by OptBayesExpt. In this example, the model function is a
    Lorentzian peak.

    Args:
        sets: A tuple of setting values, or a tuple of settings arrays
        pars: A tuple of parameter arrays or a tuple of parameter values
        cons: A tuple of floats

    Returns:  the evaluated function
    """
    # unpack the settings
    x, = sets
    # unpack model parameters
    x0, a, b, s = pars
    # unpack model constants
    d, = cons

    # calculate the Lorentzian
    return b + a / (((x - x0) / d) ** 2 + 1)


# Define the allowed measurement settings
#
# 200 values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, 100)
# sets, pars, cons are all expected to be tuples
settings = (xvals,)

# Define the prior probability distribution of the parameters
#
# resonance center x0 -- a flat prior around 3
x0_min, x0_max = (2, 4)
x0_samples = rng.uniform(x0_min, x0_max, n_samples)
# amplitude parameter a -- flat prior
a_samples = rng.uniform(-400, -2000, n_samples)
# background parameter b -- a gaussian prior around 250000
b_mean, b_sigma = (50000, 1000)
b_samples = rng.normal(b_mean, b_sigma, n_samples)
sigma_samples = rng.exponential(500, n_samples)
# Pack the parameters into a tuple.
# Note that the order must correspond to how the values are unpacked in
# the model_function.
parameters = (x0_samples, a_samples, b_samples, sigma_samples)

# Define Constants
#
dtrue = .1
constants = (dtrue,)

# make an instance of OptBayesExpt
#
my_obe = OptBayesExptSweeper(my_model_function, settings, parameters,
                             constants, scale=False)

########################################################################
#           MEASUREMENT LOOP
########################################################################

# Set up measurement simulator
#
# Randomly select "true" parameter values for simulation
true_pars = tuple([np.random.choice(param) for param in parameters])
# MeasurementSimulator.simdata() provides simulated data
# See optbayesexpt/obe_utils.py.
my_sim = MeasurementSimulator(my_model_function, true_pars, constants,
                              noise_level=noise_level)

# arrays to collect the outputs
xdata = []
ydata = []
sig = []
iter_trace = []
iterations = 0
# Perform measurements
while iterations < n_measure:

    # determine settings for the measurement
    # OptBayesExpt does Bayesian experimental design
    if optimal:
        start, stop = my_obe.opt_setting()
    else:
        start, stop = my_obe.good_setting(pickiness=pickiness)

    sweep_x_values = xvals[start : stop]
    print(len(sweep_x_values))
    # simulate a measurement
    ymeasure = my_sim.simdata((sweep_x_values,))
    [xdata.append(x) for x in sweep_x_values]
    [ydata.append(y) for y in ymeasure]

    # package the results
    measurement = ((sweep_x_values,), ymeasure)
    # OptBayesExpt does Bayesian inference
    my_obe.pdf_update(measurement)

    # OptBayesExpt provides statistics to track progress
    sigma = my_obe.std()
    sig.append(sigma[0])
    iterations += len(sweep_x_values)
    iter_trace.append(iterations)

########################################################################
#          PLOTTING
########################################################################
# plotting uses matplotlib.pyplot

plt.figure(figsize=(8, 5))

# (1) plot the true curve and the "measured" data
#
# 2 rows, 2 columns, Select the first plot
ax = plt.subplot(221)
# measurement data
plt.plot(xdata, ydata, 'k.', alpha=0.2)
# true curve
truecurve = my_sim.simdata(settings, noise_level=0)
xvals, = settings
plt.plot(xvals, truecurve, 'r-')
plt.xlabel("X value")
plt.ylabel("Signal")
plt.text(.02, .9, "(a)", transform=ax.transAxes)
xlims = plt.xlim()

# (2) plot a histogram showing where measurements are concentrated
#
# three rows, one column, third plot (below the first)
ax = plt.subplot(223)
plt.hist(xdata, bins=40)
plt.text(.02, .9, "(c)", transform=ax.transAxes)
plt.ylabel("x density")
plt.xlabel("X value")
plt.xlim(xlims)

# (3) plot the evolution of the standard deviation of the center, x0
#
# 2 rows, 2 columns, 2nd plot
ax = plt.subplot(222)
plt.loglog(iter_trace, sig)
plt.text(.02, .9, "(b)", transform=ax.transAxes)
plt.ylabel("std of center")
plt.xlabel("No. of measurements")

# (4) plot the evolution of the settings, x
#
# 2 rows, 2 columns, 4th plot
ax = plt.subplot(224)
plt.plot(np.arange(len(xdata)) + 1, xdata, 'k.', alpha=.2)
plt.text(.02, .9, "(d)", transform=ax.transAxes)
plt.plot()
plt.ylabel("X value")
plt.xlabel("No. of measurements")

plt.tight_layout()
plt.show()
