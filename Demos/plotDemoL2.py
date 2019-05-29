import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

def readobe():
    morefiles_exist=True
    i = 0
    obesigarrays=[]
    while morefiles_exist:
        runlabel = '{:03d}'.format(i)
        obefilename = 'multi_run_data/obedata'+runlabel+'.txt'
        try:
            obedata = np.loadtxt(obefilename, unpack=True)
            if i == 0:
                iteration = obedata[0]
            obesigarrays.append(obedata[2])
            i += 1
        except OSError:
            morefiles_exist=False

    obesig = np.stack(obesigarrays)
    return obesig

"""
read in the fit results.  This is awkward because the fit doesn't get recorded if there's an 
error, which happens often in the first few iterations.  The files are actually different sizes. 
"""

def readfit():
    morefiles_exist = True
    fitsigarrays = []
    fititerations = []
    i = 0
    while morefiles_exist:
        runlabel = '{:03d}'.format(i)
        fitfilename = 'multi_run_data/fitdata' + runlabel + '.txt'
        try:
            fitdata = np.loadtxt(fitfilename, unpack=True)
            fititerations.append(fitdata[0])
            fitsigarrays.append(fitdata[1])
            i += 1
        except OSError:
            morefiles_exist = False

    # Measure the space needed
    last_iteration = [a[-1] for a in fititerations]
    max_iteration = np.array(last_iteration).max()

    fitsig = np.zeros((len(fitsigarrays), int(max_iteration / 20)))
    fitmask = np.ones((len(fitsigarrays), int(max_iteration / 20)))

    # fill in the mask and fitsig
    for i, iterations in enumerate(fititerations):
        indices = np.array(iterations / 20 - 1, dtype='int')
        # validate entries where the index is present
        fitmask[i, indices] = 0
        fitsig[i, indices] = fitsigarrays[i]
        # invalidate outliers
        # toobig = np.argwhere(fitsig[i] > 3.0)
        # fitmask[i,toobig.flatten()] = 1
        toobig = fitsig[i] > 3
        fitmask[i] = np.logical_or(fitmask[i], toobig)
        # invalidate nans
        isnan = np.isnan(fitsig[i])
        fitmask[i] = np.logical_or(fitmask[i], isnan)

    fitsig_masked = ma.masked_array(fitsig, mask=fitmask)
    return fitsig_masked

def plotsigmas(ax1, obesig, fitsig_masked):
    obe_iterations = np.arange(obesig.shape[1]) + 1
    fit_iterations = np.arange(0, fitsig_masked.shape[1])*20 + 20

    obe_sigmean = obesig.mean(axis=0)
    fit_sigmean = fitsig_masked.mean(axis=0)

    # calculate the 20th and 80th percentile of the sigma values for each iteration count.
    obesig2080 = np.array([ np.percentile(a, (20,80)) for a in obesig.T])
    fitsigt = fitsig_masked.transpose()
    fitsig2080 = np.array([ np.percentile(a[~a.mask], (20,80)) for a in fitsigt])

    # plots
    # plt.rc('font', size=14)
    # plt.rc('mathtext', fontset='stixsans')
    #
    # ax1 = plt.axes()
    #
    ax1.plot(obe_iterations, obe_sigmean, c='blue', label='OptBayesExpt')
    ax1.plot(fit_iterations, fit_sigmean, c='red', label='average & fit')
    ax1.fill_between(obe_iterations, obesig2080[:,0], obesig2080[:,1], color='blue', alpha=0.3)
    ax1.fill_between(fit_iterations, fitsig2080[:,0], fitsig2080[:,1], color='red', alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(0.005, 1)
    ax1.set_xlabel('Measurement count')
    ax1.set_ylabel('Uncertainty')
    ax1.legend()
    ax1.arrow(500, .014, 4500, 0, facecolor='k', edgecolor='None', zorder=30)
    ax1.text(1000, .015, "x 10")
    return ax1


if __name__ == '__main__':
    obesig = readobe()
    fitsig_masked = readfit()
    axes = plt.axes()
    axes = plotsigmas(axes, obesig, fitsig_masked)

    plt.tight_layout()
    plt.show()


