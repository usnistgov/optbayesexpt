"""
Pi pulse tuner

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from optbayesexpt import OptBayesExpt

myOBE = OptBayesExpt()
"""
ESTABLISH THE EXPERIMENTAL MODEL
"""


# this is the model of our experiment
def rabicounts(pulsetime, delta_f, B1, f_center, baseline, contrast, T1):
    """
    A model of Rabi chevrons
    :param pulsetime:  Duration of microwave pulse
    :param delta_f:    detuning relative to a reference frequency
    :param B1:         Rabi frequency
    :param f_center:   true center of resonance relative to reference frequency
    :param baseline:   background signal
    :param contrast:   Maximum fractional change in signal for pi pulse
    :param T1:         coherence time
    :return:
    """
    zz = ((delta_f - f_center) / B1)**2
    f_rabi = np.hypot(delta_f - f_center, B1)
    return baseline*(1 - np.exp(-pulsetime / T1)*contrast / 2 *
                     (1 - np.cos(np.pi * 2 * f_rabi * pulsetime)) / (zz + 1))


def my_model_function(sets, pars, cons):

    # unpack the experimental settings
    pulsetime = sets[0]
    delta_f = sets[1]

    # unpack model parameters
    B1 = pars[0]
    f_center = pars[1]

    # unpack model constants
    # N/A
    baseline = cons[0]
    contrast = cons[1]
    T1 = cons[2]

    return rabicounts(pulsetime, delta_f, B1, f_center, baseline, contrast, T1)


myOBE.model_function = my_model_function

"""
settings and parameter and constants
"""
# define the measurement setting space
# 101 delay times up to 1 us
pulsetime = np.linspace(0, 1, 101)
# 101 detunings (MHz)
detune = np.linspace(-10, 10, 101)

# tell it to the BOE
myOBE.sets = (pulsetime, detune)

# define the parameter space where the peak could be found
# Rabi freuquency
B1min = 1
B1max = 5
B1 = np.linspace(B1min, B1max, 71)
fc_min = -7
fc_max = 7
f_center = np.linspace(fc_min, fc_max , 71)
# baseline = np.linspace(50000, 60000, 51)
# contrast = np.linspace(.05, .15, 11)
myOBE.pars = (B1, f_center)
param_extent=(B1min, B1max, fc_min, fc_max)

baseline = 100000
contrast = .01
T1 = .5
myOBE.cons = (baseline, contrast, T1)

# Settings, parameters, constants and model all defined, so set it all up
myOBE.config()

# put in a prior
B1prior = np.exp(-(B1-3.0)**2/2/2.0**2)
fcprior = np.exp(-(f_center)**2/2/4.0**2)

myOBE.set_pdf(probvalarrays=(B1prior, fcprior))

"""
MEASUREMENT SIMULATION
"""

# secret stuff - to be used only by the measurement simulator
# pick the parameters of the true resonance
fc_true = (fc_max-fc_min) * np.random.rand() + fc_min  # pick a random resonance center
B1_true = (B1max - B1min) * np.random.rand() + B1min  # pick a random background


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
    global Nmeasure
    global pickiness
    global optimum
    global noiselevel
    # global drawplots

    ptdata = np.zeros(Nmeasure)
    dfdata = np.zeros(Nmeasure)
    bmeandata = np.zeros(Nmeasure)
    dfmeandata = np.zeros(Nmeasure)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=.25)

    gs = gridspec.GridSpecFromSubplotSpec(2, 3, ax2, hspace=0.05, wspace=0.05)
    pdf_milestones = [0, 1, 2, 5, 20, 50]
    gridlist = []
    pdf_marker = 0

    for i in np.arange(Nmeasure):
        if optimum:
            ptset, dfset = myOBE.opt_setting()
        else:
            ptset, dfset = myOBE.good_setting(pickiness=pickiness)
        ptdata[i] = ptset
        dfdata[i] = dfset

        if i == pdf_milestones[pdf_marker]:
            zpdf = myOBE.get_pdf().transpose()
            ax = plt.subplot(gs[int(pdf_marker/3), pdf_marker % 3])
            gridlist.append(ax)
            plt.imshow(zpdf, origin='bottom', extent=param_extent, aspect='auto',
                       vmin=0, cmap='cubehelix')
            plt.text(1.5, 5, "N = {}".format(i), color='orange')
            plt.axis('off')
            pdf_marker += 1

        measurement = simdata(ptset, dfset)

        myOBE.pdf_update((ptset, dfset), measurement, np.sqrt(measurement))

        bmean, bsig = myOBE.get_mean(0)
        dfmean, dfsig = myOBE.get_mean(1)
        bmeandata[i] = bmean
        dfmeandata[i] = dfmean
        # if i % 10 == 0:
        #     print(i, 'B1 = {:5.3f} $\pm$ {:5.3f};  df = {:5.3f} $\pm$ {:5.3f}'.format(bmean, bsig,
        #            dfmean, dfsig))



    print('B1_true = {:6.3f}; df_true = {:6.3f}'.format(B1_true, fc_true))

    ytrue = rabicounts(*myOBE.allsettings, B1_true, fc_true, baseline, contrast, T1)
    extent = (pulsetime[0], pulsetime[-1], detune[0], detune[-1])

    plt.sca(ax1)
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('$\Delta$f (MHz)')
    plt.imshow(ytrue.transpose(), origin='bottom', extent=extent, aspect='auto',
               cmap='cubehelix', vmin=99000)
    plt.colorbar(ticks=[99000, 100000])
    plt.scatter(ptdata, dfdata, s=9, c=np.arange(len(ptdata)), cmap='Reds')

    plt.sca(gridlist[3])
    plt.axis('on')
    plt.xlabel('Rabi Frequency (MHz)')
    plt.ylabel('detuning (MHz)')


    plt.show()


Nmeasure = 51

pickiness = 4
optimum = True

batchdemo()

# import cProfile
# cProfile.run('batchdemo()', sort='cumtime')
