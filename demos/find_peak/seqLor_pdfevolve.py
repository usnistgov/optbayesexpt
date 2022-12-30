"""
Provides an animation of the measurement process for a Lorentzain peak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import optbayesexpt as obe


def lorentz(x, x0, a, b, d):
    """this is the model of our experiment - a Lorentzian peak.

    Calculate a Lorentzian function of x
    All parameters may be scalars or they may be arrays
    - as long as the arrays interact nicely

    Args:
        x (float or array): measurement setting
        x0 (float or array): peak center value parameter
        a (float or array): Amplitude parameter
        b (float or array): background parameter
        d (float or array): half-width at half-max parameter

    Returns: y  model output (float)
    """
    return b + a / (((x - x0) * 2 / d) ** 2 + 1)


def my_model_function(settings, parameters, constants):
    # a wrapper for the Lorentz function
    # unpack the experimental settings
    x, = settings
    # unpack model parameters
    x0, a, b, d = parameters
    # unpack model constants
    # d = cons[0]

    return lorentz(x, x0, a, b, d)


"""
Settings and parameters and constants
"""
# define the measurement setting space
# 50 values between 1.5 and 4.5 (GHz)
xvals = np.linspace(1.5, 4.5, 100)

# sets, pars, cons are all expected to be tuples
sets = (xvals,)

# define the prior probability distribution of parameters
# represented by n_samples samples.
n_samples = 10000

# resonance center x0 around 3

x0_min, x0_max = (2, 4)
x0_samples = np.random.uniform(x0_min, x0_max, n_samples)

# amplitude parameter A
A_samples = np.random.uniform(0, 2, n_samples)

# background parameter B
B_mean, B_sigma = 0, .5
Blimits = (B_mean - B_sigma, B_mean + B_sigma)
B_samples = np.random.uniform(*Blimits, n_samples)

# full width half max linewidth
d_samples = np.random.uniform(.02, .5, n_samples)
# Pack the parameters into a tuple
# the order must correspond to unpacking in the model_function
pars = (x0_samples, A_samples, B_samples, d_samples)

# define the known constants
cons = ()

"""
Create an instance of the OptBayesExpt class for our use
"""
myOBE = obe.OptBayesExpt(my_model_function, sets, pars, cons, scale=False,
                         n_draws=30)

"""
MEASUREMENT SIMULATION
"""
# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
x0true = np.random.choice(x0_samples)  # pick a random resonance x0
Atrue = np.random.choice(A_samples)  # pick a random Amplitude
Btrue = np.random.choice(B_samples)  # pick a random Background
dtrue = np.random.choice(d_samples)  # pick a random width
true_pars = (x0true, Atrue, Btrue, dtrue)
noise_level = 1.0

sim = obe.MeasurementSimulator(my_model_function, true_pars, cons,
                                   noise_level=noise_level)

""" 
THE ANIMATION 
"""
ax1, ax2, ax3 = (None, None, None)


def plotinit():
    """  Called for an initial plot to create the background and set scales,
    labels, etc. Data will get replaced in the animation
    """
    global fig, line1, line3, line4, line5, line6
    global ax1, ax2, ax3

    # create 3 axes to house the plots
    spacing = 0.005
    margin = 0.12
    bottom, height = margin, .5 - spacing / 2 - margin
    left, width = margin, .45 - spacing / 2 - margin
    right = .5 + spacing / 2 + margin

    leftrect = [left, bottom, width, height]
    rightrect = [right, bottom, width, height]
    toprect = [left, margin + height + spacing + margin,
               .95 - left, .95 - (margin + height + spacing + margin)]

    # Somehow, the ax* were getting created twice, triggering warnings
    if ax1 is None:
        ax1 = fig.add_axes(toprect)
    if ax2 is None:
        ax2 = fig.add_axes(leftrect)
    if ax3 is None:
        ax3 = fig.add_axes(rightrect)

    # Plot the "true curve" and data
    ytrue = lorentz(xvals, x0true, Atrue, Btrue, dtrue)
    ax1.set_xlim(xvals.min(), xvals.max())
    ax1.set_ylim(-2, 3)
    ax1.set_xlabel("x value")
    ax1.set_ylabel("y value")
    ax1.set_title('x0 = {:4.2f}, d = {:4.3f}, A = {:4.2f}, '
                  'B = {:4.2f}'.format(x0true, dtrue, Atrue, Btrue))
    # no data yet
    line1, = ax1.plot([], [], 'k.', markersize=5.0, alpha=.2)
    # true curve
    line2, = ax1.plot(xvals, ytrue, 'r-', linewidth=2)

    # pdf data
    pdf = myOBE.parameters[:, :n_plot]
    weights = np.zeros(n_plot)

    # Code weights as opacity through the alpha channel
    # make array of (0, 0, 0, alpha)
    color = np.zeros(weights.shape + (4,))
    color[:, -1] = weights * n_samples * myOBE.tuning_parameters[
        'resample_threshold'] / 10

    line3 = ax2.scatter(pdf[0], pdf[3], c=color, s=10)
    line5 = ax2.scatter(x0true, dtrue, c='red', s=10)
    ax2.set_xlim(xvals.min(), xvals.max())
    ax2.set_xlabel("peak center")
    ax2.set_ylabel("Peak width")

    line4 = ax3.scatter(pdf[2, :n_plot], pdf[1, :n_plot], c=color, s=10)
    line6 = ax3.scatter(Btrue, Atrue, c='red', s=10)
    ax3.set_xlim(*Blimits)
    ax3.set_xlabel("background")
    ax3.set_ylabel("Peak height")

    return line1, line3, line4, line5, line6


def myfunc(data):
    # a function called by the animation to replace plotted data with updates
    xdata, ydata, pdf, weights = data

    # update measured data plot
    line1.set_data(xdata, ydata)
    # lines[1].set_data(x0vals, pdf)
    color = np.zeros(weights.shape + (4,))
    rawcolor = weights * n_samples * myOBE.tuning_parameters[
        'resample_threshold'] / 10
    color[:, -1] = np.where(rawcolor > 1, 1, rawcolor)
    offsets = np.vstack((pdf[0], pdf[3])).T
    line3.set_offsets(offsets)
    line3.set_color(color)
    line5.set_offsets([x0true, dtrue])

    offsets = np.vstack((pdf[2], pdf[1])).T
    line4.set_offsets(offsets)
    line4.set_color(color)
    line6.set_offsets([Btrue, Atrue])

    return line1, line3, line4, line5, line6


def myframes():
    # a generator that comes up with new data for the next frame
    cnt = 0
    global xvals
    global pickiness
    global optimum

    xdata = []
    ydata = []
    while cnt < Nmeasure:
        cnt += 1
        if cnt % 100 == 0:
            print('Iteration {}'.format(cnt))
        """ Get the new measurement setting by Bayes Optimization  """
        if optimum:
            xmeas = myOBE.opt_setting()
        else:
            xmeas = myOBE.good_setting(pickiness=pickiness)
        ymeas = sim.simdata(xmeas)
        """ report the measurement back in order to update """
        result = (xmeas, ymeas, noise_level)
        myOBE.pdf_update(result)

        xdata.append(xmeas)
        ydata.append(ymeas)
        mypdf = myOBE.parameters[:, :n_plot]
        weights = myOBE.particle_weights[:n_plot]
        # sleep(.1)

        # genearator yields updated data
        yield xdata, ydata, mypdf, weights


def livedemo():
    global fig

    fig = plt.figure(figsize=(8, 6))
    ani = animation.FuncAnimation(fig, myfunc, frames=myframes,
                                  init_func=plotinit, blit=True, interval=0,
                                  repeat=False)
    plt.show()


if __name__ == "__main__":
    plt.rc('font', size=14)
    Nmeasure = 1000
    n_plot = 500
    # optimum = True
    optimum = False
    pickiness = 10

    livedemo()
