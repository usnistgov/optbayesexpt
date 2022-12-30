"""
This ``numbaLorentzian.py`` script demonstrates the use of ``numba`` and its
@njit decorator to accelerate the more numerics-intensive functitons in
optbayesexpt. The purpose of ``numba`` is to pre-compile functions
just-in-time (jit) so that when the function is called, the CPU runs machine
code without involvement from the Python interpreter.

Boolean variable ``use_jit`` enables (``use_jit = True``) or disables (
``use_jit = False``) pre-compilation.  The ``use_jit`` variable determines
whether the ``@njit`` decorator is used in ``_my_model_calc()``.
``use_jit`` is also passed as a keyword argument to OptBayesExpt to
enable/disable pre-compilation of selected functions in OptBayesExpt and
ParticlePDF classes.

The output of ``numbaLorentzian.py`` includes a code profile that shows
where the computation time was spent.  Important column headings include:

 - 'tottime' sorts the output by time spent in a function, not counting
   called functions.
 - 'cumtime' includes time spent on called functions.
   '30' limits the number of lines of output
"""

import numpy as np
import matplotlib.pyplot as plt
from optbayesexpt import OptBayesExpt, MeasurementSimulator
import cProfile
from pstats import Stats

#####################################################
# One of the following 2 line should be uncommented
use_jit = True
# use_jit = False
#####################################################

GOT_NUMBA = True
try:
    from numba import njit, float64
except ImportError as msg:
    GOT_NUMBA = False   # and that's a problem.
    print(msg)
    print("The numbaLorentzian.py demo requires the numba package.")
    print("Hint:")
    print("> pip install numba")
    print(" -- or, for the Anaconda distribution --")
    print("> conda install numba")
    exit(1)

########################################################################
#           SETUP
########################################################################

# Script tuning parameters
#
# Measurement simulation: added noise level
noise_level = 500
# Measurement loop: Quit measuring after ``n_measure`` measurement iterations
n_measure = 3000

# random number generator
try:
    rng = np.random.default_rng()
except:
    rng = np.random

# Tuning the OptBayesExpt behavior
#
# The parameter probability distribution is represented by ``n_samples``
# samples from the distribution.
n_samples = 50000
# The parameter selection method is determined by ``optimal``.
# optimal = True        # use OptBayesExpt.opt_setting()
optimal = False  # use OptBayesExpt.good_setting() with pickiness
pickiness = 19  # ignored when optimal == True


# Describe how the world works with a model function
#
if use_jit:
    @njit(cache=True)
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
        x0, a, b = pars
        # unpack model constants
        d, = cons

        # calculate the Lorentzian
        # return _my_model_calc(a, x, x0, b, d)
        return a / ((2 * (x - x0) / d)**2 + 1) + b
else:
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
        x0, a, b = pars
        # unpack model constants
        d, = cons

        # calculate the Lorentzian
        # return _my_model_calc(a, x, x0, b, d)
        return a / ((2 * (x - x0) / d) ** 2 + 1) + b

# Define the allowed measurement settings
#
# 200 values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, 200)
# sets, pars, cons are all expected to be tuples
settings = (xvals,)

# Define the prior probability distribution of the parameters
#
# resonance center x0 -- a flat prior around 3
x0_min, x0_max = (2, 4)
x0_samples = rng.uniform(x0_min, x0_max, n_samples)
# amplitude parameter a -- flat prior
a_samples = rng.uniform(-2000, -400, n_samples)
# background parameter b -- a gaussian prior around 250000
b_mean, b_sigma = (50000, 1000)
b_samples = rng.normal(b_mean, b_sigma, n_samples)
# Pack the parameters into a tuple.
# Note that the order must correspond to how the values are unpacked in
# the model_function.
parameters = (x0_samples, a_samples, b_samples)
param_labels = ['Center', 'Amplitude', 'Background']
# Define Constants
#
dtrue = .1
constants = (dtrue,)

# make an instance of OptBayesExpt
#
my_obe = OptBayesExpt(my_model_function, settings, parameters, constants,
                      scale=False, use_jit=use_jit)

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
xdata = np.zeros(n_measure)
ydata = np.zeros(n_measure)
sig = np.zeros((n_measure, 3))

# Perform measurements using cProfile to show where computation time goes
with cProfile.Profile() as profiler:
    for i in np.arange(n_measure):

        # determine settings for the measurement
        # OptBayesExpt does Bayesian experimental design
        if optimal:
            xmeas = my_obe.opt_setting()
        else:
            xmeas = my_obe.good_setting(pickiness=pickiness)

        # simulate a measurement
        ymeasure = my_sim.simdata(xmeas)
        xdata[i] = xmeas[0]
        ydata[i] = ymeasure

        # package the results
        measurement = (xmeas, ymeasure, noise_level)
        # OptBayesExpt does Bayesian inference
        my_obe.pdf_update(measurement)

        # OptBayesExpt provides statistics to track progress
        sigma = my_obe.std()
        sig[i] = sigma

        # entertainment
        if i % 100 == 0:
            print("{:3d}, sigma = {}".format(i, sigma[0]))

# print out timing profile data
stats = Stats(profiler)
# 'tottime' sorts the output by time spent in a function, not counting
# called functions.
# 'cumtime' includes time spent on called functions.
# '30' limits the number of lines of output
stats.strip_dirs().sort_stats('tottime').print_stats(30)

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
plt.text(0.02, .9, '(a)', transform=ax.transAxes)
xlims = plt.xlim()

# (2) plot a histogram showing where measurements are concentrated
#
# three rows, one column, third plot (below the first)
ax = plt.subplot(223)
plt.hist(xdata, bins=40)
plt.ylabel("x density")
plt.xlabel("X value")
plt.text(0.02, .90, '(c)', transform=ax.transAxes)
plt.xlim(xlims)

# (3) plot the evolution of the standard deviation of the center, x0
#
# 2 rows, 2 columns, 2nd plot
ax = plt.subplot(222)

# for label, sigtrace in zip(param_labels, sig.T):
#     plt.loglog(np.arange(n_measure) + 1, sigtrace/sigtrace[0], label=label)
# plt.legend()
plt.loglog(np.arange(n_measure) + 1, sig[:, 0])
plt.ylabel("std of center")
plt.xlabel("No. of measurements")
plt.text(0.02, .8, '(b)', transform=ax.transAxes)

# (4) plot the evolution of the settings, x
#
# 2 rows, 2 columns, 4th plot
ax = plt.subplot(224)
plt.plot(np.arange(n_measure) + 1, xdata, 'k.', alpha=.2)
plt.plot()
plt.ylabel("X value")
plt.xlabel("No. of measurements")
plt.text(0.02, .9, '(d)', transform=ax.transAxes)

plt.tight_layout()
plt.show()
