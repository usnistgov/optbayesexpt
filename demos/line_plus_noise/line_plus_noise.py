"""
This script simulates measurements of a linear function, allowing
several demonstrations:

- Use of the OptBayesExptBase class for customization.  Code for the
  customized class is in ``obe_line_plus_noise.py``
- Including measurement uncertainty as an "unknown" parameter. To include
  uncertainty, we customize likelihood() and yvar_noise_model() methods.
- Comparing setting selection strategies. The output plots compare a run
  using opt_setting() with runs using good_setting() and various pickiness
  parameters.

"""
import numpy as np
import matplotlib.pyplot as plt
from optbayesexpt import MeasurementSimulator, OptBayesExptNoiseParameter

# Script tuning parameters
#
# the parameter probability distribution is represented by samples from the
# distribution.
n_samples = 50000
# quit measuring after n_measure measurement iterations
n_measure = 100
# random number generator
try:
    rng = np.random.default_rng()
except AttributeError:
    rng = np.random

########################################################################
#           OptBayesExpt SETUP
########################################################################


def model_function(settings, parameters, constants):
    """Evaluates a linear experimental model

    Args:
        settings: one or many combinations of setting values
        parameters: many or one combinations of parameter values
        constants: Floats that might change occasionally

    Returns: float, y = m * x + b

    """
    # unpack the experimental settings
    x, = settings
    # unpack model parameters
    m, b, sigma = parameters
    # note that sigma is unpacked from parameters, but it isn't used in
    # the model calculation.

    return m * x + b

# define the measurement settings
#
# 101 possible x values
xsettings = np.linspace(0, 1, 101)
#  packaged as a tuple
sets = (xsettings,)

# define the parameter space
#
mvals = np.random.uniform(-1, 1, n_samples)
bvals = np.random.uniform(-1, 1, n_samples)
sigs = np.random.exponential(.1, n_samples)
# package as a tuple
pars = (mvals, bvals, sigs)

# define the constants - even though there aren't any.
#
cons = ()

# create an instance
#
my_obe = OptBayesExptNoiseParameter(model_function, sets, pars, cons,
                                    scale=False,
                                    noise_parameter_index=2)

########################################################################
#           MEASUREMENTS
########################################################################

# measurement simulator
#
m_true = rng.choice(mvals)  # pick a random slope betw. -1 and 1
b_true = rng.choice(bvals)  # pick a random intercept
s_true = .2
true_pars = (m_true, b_true, s_true)
# MeasurementSimulator.simdata() borrows the model function from
# my_obe and provides simulated data. See optbayesexpt/obe_utils.py.
my_sim = MeasurementSimulator(my_obe.model_function, true_params=true_pars,
                              cons=cons, noise_level=s_true)


def bayesrun(n_measure, optimum, pickiness):
    """Performs a simulated experimental run

    Args:
        n_measure (int): the number of iterations in the run
        optimum (bool):  find optimal setting (True) or a good setting (False)
        pickiness (int): Tuning parameter for selectivity of good settings.

    Returns:
        A tuple of measurement and 'fit' results

    """
    global my_obe
    global pars
    my_obe.set_pdf(pars)

    # create a place for settings and measurement results
    xdata = np.zeros(n_measure)
    ydata = np.zeros(n_measure)

    # the measurement loop
    for i in np.arange(n_measure):
        # get a recommended setting
        if optimum:
            xset = my_obe.opt_setting()
        else:
            xset = my_obe.good_setting(pickiness=pickiness)
        xdata[i], = xset

        # simulated measurement
        ymeas = my_sim.simdata(xset)
        ydata[i] = ymeas

        # report back to OBE
        measurement_result = (xset, ymeas)
        my_obe.pdf_update(measurement_result)

    # Extract statistics from the final distributions
    means = my_obe.mean()
    sigs = my_obe.std()
    ytrue = my_sim.simdata(sets, true_pars, noise_level=0)
    ymean = my_sim.simdata(sets, means, noise_level=0)

    return xdata, ydata, ytrue, ymean, means, sigs


def batchplot(subplot, xdata, ydata, ytrue, ymean, means, sigs):
    """Creates histogram and graphs in a subplot

    The args include the target subplot axes and the outputs of bayesrun()

    Args:
        subplot: The destination axes
        xdata: X coordinates of measured points
        ydata: Y coordinates of measured points
        ytrue: Y values of true curve
        ymean: Y values of "best fit" curve
        means: best fit parameters
        sigs: Uncertainty of parameters

    Returns: None
    """

    global top, bottom

    # create 2nd y axis
    ax_l = subplot.twinx()
    # It seems that matplotlib wants to plot the original axes first,
    # and the twinx()-defined axes 2nd.  So, to get the histogram to appear
    # behind the plotted data, we plot it in the original axes but move the
    # y-axis.

    # Histogram with y-axis on the right
    subplot.hist(xdata, bins=20, color='lightblue')
    # force subplot's y axis to the right
    subplot.yaxis.tick_right()
    subplot.yaxis.set_label_position('right')
    subplot.set_ylabel("points")
    subplot.set_xlabel('x')
    ymax = n_measure / 2 * 1.1
    subplot.set_ylim(0, ymax)

    # data and curves with y-axis on the left
    ax_l.plot(xsettings, ytrue, 'r-', label='True')
    ax_l.plot(xsettings, ymean, 'b-', label='Est.')
    ax_l.plot(xdata, ydata, 'k.', alpha=.2)
    # force ax_l's y axis to the left
    ax_l.yaxis.tick_left()
    ax_l.yaxis.set_label_position('left')
    ax_l.set_ylim((bottom, top))
    ax_l.set_ylabel('y')

    # annotate plots with results
    y1 = bottom + 0.93 * (top - bottom)
    y2 = bottom + 0.85 * (top - bottom)
    y3 = bottom + 0.77 * (top - bottom)
    m_mean, b_mean, s_mean = means
    m_sig, b_sig, s_sig = sigs
    plt.text(0.05, y1, 'm = {:5.3f}$\pm${:5.3f}'.format(m_mean, m_sig))
    plt.text(0.05, y2, 'b = {:5.3f}$\pm${:5.3f}'.format(b_mean, b_sig))
    plt.text(0.05, y3, '$\sigma$ = {:5.3f}$\pm${:5.3f}'.format(s_mean, s_sig))
    plt.legend(loc=8)

    return subplot.yaxis, ax_l.yaxis


########################################################################
#           MEASURE & PLOT
########################################################################


print("true parameters = ", true_pars)
# Four subplots for different script parameters
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
line_limits = [b_true, b_true + m_true]
top = max(line_limits) + .5
bottom = min(line_limits) - .5

results = bayesrun(n_measure, optimum=True, pickiness=0)
batchplot(axes[0], *results)
plt.title('opt_setting()')

pickiness = 15
results = bayesrun(n_measure, optimum=False, pickiness=pickiness)
batchplot(axes[1], *results)
plt.title('good_setting(pickiness={})'.format(pickiness))

pickiness = 5
results = bayesrun(n_measure, optimum=False, pickiness=pickiness)
batchplot(axes[2], *results)
plt.title('good_setting(pickiness={})'.format(pickiness))

pickiness = 1
results = bayesrun(n_measure, optimum=False, pickiness=pickiness)
batchplot(axes[3], *results)
plt.title('good_setting(pickiness={})'.format(pickiness))

plt.tight_layout()
plt.show()
