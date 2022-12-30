"""
Simulates measurement of an inductor coil using a lockin amplifier.

This demo features a custom child class of OptBayesExptBase with several
features not covered by OptBayesExpt:

    - Multiple (two) measurement outputs. In this case, the X and Y
      channels of a lockin amplifier, or equivalently the I and Q channels
      of phase-sensitive detection. See
      ``self.VALUES_PER_MEASUREMENT``, ``yvar_from_parameter_draws()``
      ``Likelihood()``,
    - Setting-dependent cost function. Extra time is required for settling
      when the setting is changed.  See ``cost_estimate()`` and
      ``self.cost_of_changing_setting``.
    - Unknown noise level.  The standard deviation of the (Gaussian)
      measurement noise is one of the parameters.  See
      ``likelihood()`` and ``y_var_noise_model()``
    - enforcement of parameter constraints.  See ``police_parameters()``
    - Complex numbers are used in the model function.

    settings
        - excitation frequency
    parameters:
        - inductance,
        - resistance,
        - capacitance and
        - voltage noise standard deviation
    constants
        - None
"""

import numpy as np
import matplotlib.pyplot as plt
from optbayesexpt import MeasurementSimulator, trace_sort
from optbayesexpt import OptBayesExptNoiseParameter

# script parameters
#
# number of iterations of the measurement loop
n_measure = 500
# number of samples in the distribution
n_samples = 50000
# parameter selection strategy
optimal = True          # if optimal settings are needed
# optimal = False,          # if good settings are needed
pickiness = 1            # selectivity parameter, ignored for optimal == True
# measurement cost
# changing the setting requires 5 t
cost_of_moving = 5
# using numpy's new style of random number generator
try:
    rng = np.random.default_rng()
except AttributeError:
    # ... or not
    rng = np.random


########################################################################
#          SETUP
########################################################################


def coil_model(sets, pars, cons):
    """ Evaluates a model of the experiment

    Evaluates the equivalent of a fit function. The argument structure is
    required by OptBayesExpt.

    Args:
        sets: A tuple of setting values, or a tuple of settings arrays
        pars: A tuple of parameter arrays or a tuple of parameter values
        cons: An array of 2 floats or an array of 2 arrays.

    Returns:  the evaluated function

    For the coil measurement, the model is the coil inductance in series
    with the wire resistance, all in parallel with the turn-to-turn
    capacitance.

    o                          o
    |      R            L      |
    ---/\/\/\/\-----CCCCCCCC----  1
    |                          |
    ------------| |-------------  2
                 C
    """
    # unpack the settings
    w, = sets
    # unpack model parameters
    # noise_sigma is not used here.
    L, R, C, noise_sigma = pars
    # unpack model constants
    # None
    # Admittance (inverse of impedance) of branch 1
    Y1 = 1 / (R + 1j * w * L)
    # Admittance of branch 2
    Y2 = 1j * w * C
    # Impedance = inverse of total conductance
    Z = 1 / (Y1 + Y2)

    # Returning an array of 2 values, the real and imaginary parts
    return np.array((np.real(Z), np.imag(Z)))


# define a subclass of OptBayesExptNoiseParameter with a
# enforce_parameter_constraints() method for all parameters
class OptBayesExptLockinCleanParams(OptBayesExptNoiseParameter):

    def __init__(self, coil_model, sets, params, cons,
                 cost_of_changing_setting=1.0, **kwargs):
        OptBayesExptNoiseParameter.__init__(self, coil_model, sets, params,
                                           cons, **kwargs)
        self.cost_of_changing_setting = cost_of_changing_setting

    def enforce_parameter_constraints(self):
        """
        All of the coil parameters and noise values must be > 0.  Assign
        zero probability to any violators.
        """
        changes = False
        for param in self.parameters:
            # find the violators
            bad_ones = np.argwhere(param < 0).flatten()
            if len(bad_ones) > 0:
                changes = True
                for violator in bad_ones:
                    # effective death penalty.  Next resample will remove
                    self.particle_weights[violator] = 0

        if changes is True:
            # rescale the particle weights
            self.particle_weights = self.particle_weights \
                                    / np.sum(self.particle_weights)

    def cost_estimate(self):
        """
        Estimate the cost of measurements, depending on settings

        The denominator of the *utility* function allows measurement
        resources (e.g. setup time + data collection time) to be entered
        into the utility calculation.

        Returns:
            :obj:`float`, otherwise an :obj:`ndarray` describing how
                measurement variance depends on settings.
        """
        setting = self.last_setting_index
        cost = np.ones_like(self.allsettings[0]) * \
               self.cost_of_changing_setting
        cost[setting] = 1.0

        return cost
    # End of class definition


# prepare settings
#
# 200 frequency values 100 Hz and 1 MHz, logarithmic spacing
frequency = np.logspace(2, 6, 200)
print(frequency[0], frequency[-1])
# all of the model calculations will use angular frequency, omega
omega = np.pi * 2 * frequency
sets = (omega,)

# prepare parameters
#
# Inductance on the scale of 1 mH
L_typical = 0.001
L_samples = rng.exponential(L_typical, n_samples)
# Coil resistance on the scale of 1 Ohm
R_typical = 10  # Ohms
R_samples = rng.exponential(R_typical, n_samples)
# Capacitance on the scale of 10 uF
C_typical = 10e-6
C_samples = rng.exponential(C_typical, n_samples)
# noise on the scale of .1 Ohm
sigma_typical = 10
sigma_samples = rng.exponential(sigma_typical, n_samples)
# Pack the parameters into a tuple.
# Note that the order must correspond to how the values are unpacked in
# the model_function.
params = (L_samples, R_samples, C_samples, sigma_samples)

# prepare constants
#
# No constants but supply an empty tuple
cons = ()

coil_obe = OptBayesExptLockinCleanParams(coil_model, sets, params, cons,
                                scale=False, noise_parameter_index=(3,3),
                                cost_of_changing_setting=cost_of_moving)
# Here, scale=False is a keyword argument that the class definition lumps
# into **kwargs and passes to OptBayesExptNoiseParam, which passes it to
# OptBayesExpt, which passes it to ParticlePDF.  ParticlePDF has a ``scale``
# argument that gets set to ``False``.
# The noise_parameter_index parameter is passed to
# OptBayesExptNoiseParameter to identify which parameter describes the
# standard deviation of experimental noise.




########################################################################
#           MEASUREMENT LOOP
########################################################################

# Set up measurement simulator
#
# Randomly select "true" parameter values for simulation
true_pars = tuple([np.random.choice(param) for param in params])
# MeasurementSimulator.simdata() provides simulated data
# See optbayesexpt/obe_utils.py.
my_sim = MeasurementSimulator(coil_obe.model_function, true_pars, cons,
                              noise_level=true_pars[-1])

# arrays to collect the outputs
frequency_trace = np.zeros(n_measure)
x_trace = np.zeros(n_measure)
y_trace = np.zeros(n_measure)

for i in np.arange(n_measure):
    # get the measurement setting. opt_setting and good_setting return
    # tuples.
    if optimal:
        wmeas = coil_obe.opt_setting()
    else:
        wmeas = coil_obe.good_setting(pickiness=pickiness)
    # simulate a measurement
    ymeasure = my_sim.simdata(wmeas)
    frequency_trace[i] = wmeas[0] / 2 / np.pi
    x_trace[i] = ymeasure[0]
    y_trace[i] = ymeasure[1]

    # report the results -- the learning phase
    measurement = (wmeas, ymeasure)
    coil_obe.pdf_update(measurement)

    # entertainment
    if i % 100 == 0:
        print("iteration {:3d}".format(i))

# produces frequency_trace, x_trace, y_trace

#########################################################
#       OUTPUTS AND PLOTTING
#########################################################

# printed outputs
#
names = ["L", "R", "C", "sigma"]
scales = np.array([1e-3, 1, 1e-6, 1])
units = ["mH ", "Ohm", "uF ", "Ohm"]
means = coil_obe.mean() / scales
stds = coil_obe.std() / scales
trues = np.array(true_pars) / scales
f_string = "{}: true = {:7.3f} {}  measured = ({:7.3f} +/- {:7.3f}) {}"
for name, true, mean, std, unit in \
        zip(names, trues, means, stds, units):
    print(f_string.format(name, true, unit, mean, std, unit))

# plotting
#
truecurve = my_sim.simdata(sets, noise_level=0)

plt.figure(figsize=(8,5))

# (1) plot the true curve and the "measured" data
ax = plt.subplot(221)
plt.text(.02, .9, "(a)", transform=ax.transAxes)
# measurement data
plt.semilogx(frequency_trace, x_trace, 'r.', alpha=0.2)
plt.semilogx(frequency_trace, y_trace, 'g.', alpha=0.2)
# true curves
xvals = sets[0] / 2 / np.pi
plt.semilogx(xvals, truecurve[0], 'r-', label="Real")
plt.semilogx(xvals, truecurve[1], 'g-', label="Imag")
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Impedance ($\Omega$)")

# (2) plot the setting behavior
ax = plt.subplot(222)
plt.text(.02, .9, "(b)", transform=ax.transAxes)
plt.semilogy(frequency_trace, '.')
plt.ylabel("Frequency Setting (Hz)")
plt.xlabel("Measurement iteration")

ax = plt.subplot(223)
plt.text(0.02, .9, "(c)", transform=ax.transAxes)
# digested measurement data
x, y, std, n_of_m = trace_sort(frequency_trace, x_trace)
size = np.array(n_of_m) * 5

plt.scatter(x, y, marker='.', c='r', s=size, alpha=0.5)

x, y, std, n_of_m = trace_sort(frequency_trace, y_trace)
plt.scatter(x, y, marker='.', c='g', s=size, alpha=0.5)

# true curve
xvals = sets[0] / 2 / np.pi
plt.semilogx(xvals, truecurve[0], 'r-', label="Real")
plt.semilogx(xvals, truecurve[1], 'g-', label="Imag")
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Impedance ($\Omega$)")

thisaxes = plt.subplot(224)
plt.xticks([])
plt.yticks([])
thisaxes.set_frame_on(False)
# printed outputs
names = ["L", "R", "C", "$\sigma$"]
scales = np.array([1e-3, 1, 1e-6, 1])
units = ["mH ", "$\Omega$", "$\mu$F ", "$\Omega$"]
means = coil_obe.mean() / scales
stds = coil_obe.std() / scales
trues = np.array(true_pars) / scales
f_string = "{}: true = {:5.3f} {} <-> ({:5.3f} +/- {:5.3f}) {}"
top = .9
delta_y = .1
# delta_x = -.25
delta_x = -.1

for name, true, mean, std, unit in \
        zip(names, trues, means, stds, units):
    info = f_string.format(name, true, unit, mean, std, unit)
    plt.text(delta_x, top, info)
    top -= delta_y

plt.tight_layout()
plt.show()
