"""
Pi pulse tuner

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from optbayesexpt import OptBayesExpt

"""
ESTABLISH THE EXPERIMENTAL MODEL
"""


# this is the model of our experiment
def rabicounts(pulsetime, delta_f, B1, f_center, baseline, contrast, T1):
    """

    Args:
        pulsetime (float): Duration of microwave pulse
        delta_f (float): detuning relative to a reference frequency
        B1 (float): Rabi frequency
        f_center (float): true center of resonance relative to reference
            frequency
        baseline (float): background signal
        contrast (float): Maximum fractional change in signal for pi pulse
        T1 (float): coherence time
    """
    zz = ((delta_f - f_center) / B1)**2
    f_rabi = np.hypot(delta_f - f_center, B1)
    return baseline*(1 - np.exp(-pulsetime / T1)*contrast / 2 *
                     (1 - np.cos(np.pi * 2 * f_rabi * pulsetime)) / (zz + 1))


def my_model_function(sets, pars, cons):

    # unpack the experimental settings
    pulsetime, delta_f = sets

    # unpack model parameters
    B1, f_center = pars

    # unpack model constants
    # N/A
    baseline, contrast, T1 = cons

    return rabicounts(pulsetime, delta_f, B1, f_center, baseline, contrast, T1)


"""
settings and parameter and constants
"""
# define the measurement setting space
# 101 delay times up to 1 us
pulsetime = np.linspace(0, 1, 101)
# 101 detunings (MHz)
detune = np.linspace(-10, 10, 101)

# tell it to the BOE
sets = (pulsetime, detune)

# define the parameter space where the peak could be found
n_samples = 10000
# Rabi freuquency
B1min = 1
B1max = 5
B1 = np.random.uniform(B1min, B1max, n_samples)
fc_min = -7
fc_max = 7
f_center = np.random.uniform(fc_min, fc_max, n_samples)
# baseline = np.linspace(50000, 60000, 51)
# contrast = np.linspace(.05, .15, 11)
pars = (B1, f_center)
# param_extent=(fc_min, fc_max, B1min, B1max)

baseline = 100000
contrast = .01
T1 = .5
cons = (baseline, contrast, T1)

# Settings, parameters, constants and model all defined, so set it all up
myOBE = OptBayesExpt(my_model_function, sets, pars, cons)


"""
MEASUREMENT SIMULATION
"""
# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
fc_true = np.random.choice(f_center)
B1_true = np.random.choice(B1)


def simdata(pt, df):
    """
    simulate a measurement at pulsetime pt and detuning df
    """
    # calculate the theoretical output result
    y = rabicounts(pt, df, B1_true, fc_true, baseline, contrast, T1)
    # add noise
    global noiselevel
    s = np.sqrt(y)            # counting noise
    if type(y) == np.ndarray:
        y += s * np.random.randn(len(y))
    else:
        y += s * np.random.randn()
    return y


""" 
RUN THE "EXPERIMENT" DEMO 
"""


def batchdemo():
    global pickiness
    global optimum
    global noiselevel
    # global drawplots

    pdf_milestones = [2, 5, 10, 20, 50, 100]
    n_measure = pdf_milestones[-1] + 1
    ptdata = np.zeros(n_measure)
    dfdata = np.zeros(n_measure)
    bmeandata = np.zeros(n_measure)
    dfmeandata = np.zeros(n_measure)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=.25)

    gs = gridspec.GridSpecFromSubplotSpec(2, 3, ax2, hspace=0.05, wspace=0.05)

    pdf_marker = 0
    axlist = []
    for i in np.arange(n_measure):
        print(i)
        if optimum:
            ptset, dfset = myOBE.opt_setting()
        else:
            ptset, dfset = myOBE.good_setting(pickiness=pickiness)
        ptdata[i] = ptset
        dfdata[i] = dfset

        if i == pdf_milestones[pdf_marker]:
            zpdf = myOBE.parameters
            weights = myOBE.particle_weights
            row = int(pdf_marker/3)
            column = pdf_marker % 3
            axlist.append(plt.subplot(gs[row, column]))
            # plt.xlim(B1min, B1max)
            # plt.ylim(fc_min, fc_max)
            plt.xlim(.5, 5.5)
            plt.ylim(-8, 8)
            plt.scatter(zpdf[0], zpdf[1], 1000*weights, 'k', alpha=.05)
            plt.scatter(B1_true, fc_true, 16, c='r')
            plt.text(1.5, 5, "N = {}".format(i), color='b')
            if row < 1:
                plt.xticks([])
            if column > 0:
                plt.yticks([])
            # plt.axis('off')
            pdf_marker += 1

        y_meas = simdata(ptset, dfset)
        measurement = ((ptset, dfset), y_meas, np.sqrt(y_meas))

        myOBE.pdf_update(measurement)

        bmean, dfmean = myOBE.mean()
        bmeandata[i] = bmean
        dfmeandata[i] = dfmean

    print('B1_true = {:6.3f}; df_true = {:6.3f}'.format(B1_true, fc_true))

    ytrue = rabicounts(*myOBE.allsettings, B1_true, fc_true, baseline,
                       contrast, T1)
    extent = (pulsetime[0], pulsetime[-1], detune[0], detune[-1])
    yshape = tuple([len(arr) for arr in sets])
    ytrue_plotted = ytrue.reshape(yshape).transpose()
    plt.sca(ax1)
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('$\Delta$f (MHz)')
    plt.imshow(ytrue_plotted, origin='bottom', extent=extent, aspect='auto',
               cmap='cubehelix', vmin=99000)
    plt.colorbar(ticks=[99000, 100000])
    plt.scatter(ptdata, dfdata, s=9, c=np.arange(len(ptdata)), cmap='Reds')

    # plt.subplot(gs[1,0])

    plt.sca(axlist[3])
    plt.axis('on')
    plt.xlabel('Rabi Frequency (MHz)')
    plt.ylabel('detuning (MHz)')

    plt.show()


pickiness = 4
optimum = True
# optimum = False

batchdemo()

# import cProfile
# cProfile.run('batchdemo()', sort='cumtime')
