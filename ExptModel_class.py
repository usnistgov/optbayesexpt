__author__ = 'Bob McMichael'

import numpy as np

class ExptModel:
    """
    A class for an experimental model geared towards Bayesian optimal experimental design
    Data:
        allparams: an iterable covering all of model parameter space
        allsettings: an iterable covering all of experimental parameter space
        constants: model parameters that remain constant
    Methods:
        eval_over_all_parameters:   wrapper to evaluate the model for every possible parameter combination
        eval_aver_all_settings:  wrapper to evaluate the model for every possible setting
        model_function:  the guts of the model calculation
    """
    def __init__(self):
        pass

    def model_config(self, settings, parameters, constants):
        self.constants = constants
        self.allsettings = np.meshgrid(*settings, indexing='ij')
        self.allparams = np.meshgrid(*parameters, indexing='ij')

    def model_function(self, sets, pars, cons):
        """
        :param settings:   list of arrays or list of floats -- experimental settings
        :param parameters: list of floats or list of arrays -- model parameters
        :param constants:  list of floats  -- constants in the model
        :return: an array with dims of setting space -- or --
                 an array with dims of parameter space -- or --
                 a float    -- depending on the arguments
        """
        pass

    def eval_over_all_parameters(self, onesettingset):
        """
        evaluate the experimental model over all possible model parameters for one set of measurement settings
        :param onesettingset: list of floats -- measurement settings
        :return: array with dims of parameter space
        """
        return self.model_function(onesettingset, self.allparams, self.constants)

    def eval_over_all_settings(self, oneparamset):
        """
        evaluate the experimental model over all possible measurement settings for one set of parameters
        :param oneparamset:  list of model parameters
        :return: array with dims of setting space
        """
        return self.model_function(self.allsettings, oneparamset, self.constants)

"""
Self_test / Example
"""
def self_test():
    """
    Example Lorentzian Experimental model

    Pretend we have a spectrometer with a knob that goes from 0 to 1 in 100 steps.
    We're measuring something that we think is going to produce a Lorentzian peak.
    """
    ## define the space of settings and parameters

    # settings
    # these are settings for the spectrometer.
    # The spectrometer has a control knob that goes from 0 to 1 in 100 steps.
    xvals = np.linspace(0, 1, 101)
    # Our spectrometer could have more than one setting in general, so we pack the settings in a tuple,
    # even though it's not really necessary in this example.
    settings = (xvals,)

    # parameters
    # Model parameters that you would allow to vary in a fit to measured data.
    # in this case, a peak center (x0) and a lhwhm ine width (dx) are values I'd like to determine.
    # These parameters could take on any real value in the physical world, but here in the computer,
    # the possible values are discrete
    # peak center.  41 possible values, 0.02 precision
    x0vals = np.linspace(.1, .9, 41)
    # width.  50 possible values 0.004 precision
    dwvals = np.linspace(.01, .2, 50)
    # pack the parameters in a tuple
    parameters = (x0vals, dwvals)

    # constants
    # Model parameters that you would decide to keep constant in a fit to measured data.
    # in this case, I "know" that my peak height is 1.0.
    A = 1.0
    # Again, only one constant, but we pack it in a tuple anyways to show the general form
    constants = (A, )

    # make a model that predicts the output of our spectrometer given settings, parametes, constants
    class LorentzModel(ExptModel):
        """
        A model specifically for Lorentzian peaks, inheriting from the ExptModel class.
        """

        def __init__(self):
            ExptModel.__init__(self)

        # The ExptModel class focuses on creating different ways to evaluate the model, but "passes"
        # on defining the model function.  Here's where we make that definition
        def model_function(self, settuple, partuple, contuple):
            """
            This is a tiny piece of code, but it's a big deal from the user perspective because it's the part
            that users will need to adapt to different models. Here, the programmer defines the model
            and how the settings, parameters and constants are interpreted.
            :param settings:   tuple of instrument settings
            :param parameters: tuple of model parameters, unknowns treated as discrete variables
            :param constants:  tuple of model constants.
            :return:
            """

            # unpack our input tuples

            # experimental settings
            x = settuple[0]
            # model parameters
            x0 = partuple[0]
            dx = partuple[1]
            # constants
            A = contuple[0]

            # and now our model for the expected spectrometer output
            return A / (1 + (x - x0) ** 2 / dx ** 2)
            # OK, this is just a one-liner, but the model could be much more complex.

    # Settings, Parameters, Constants all defined
    # Model is defined
    # Next is to start calculating

    # create a model instance
    lmodel = LorentzModel()
    lmodel.model_config(settings, parameters, constants)
    print('allsettings.shape = {}'.format(lmodel.allsettings[0].shape))
    print('allparams.shape = {}'.format(lmodel.allparams[0].shape))
    # calculate for all settings (like an experimental sweep)
    oneparameter = (.4, .1)
    ytrace = lmodel.eval_over_all_settings(oneparameter)

    # calculate for all parameters ( for likelihood calc )
    onesetting = (.6,)
    pspace = lmodel.eval_over_all_parameters(onesetting)

    # Calculations done.

    #
    # plot the results
    #
    import matplotlib.pyplot as plt
    plt.figure()

    # all settings plot
    plt.subplot(211)
    plt.plot(xvals, ytrace)
    plt.xlabel('setting, x')
    plt.ylabel('model value')
    plt.title("all settings, parameters (x0=.4, dx=.1)")

    # all parameters image
    plt.subplot(212)
    extent = [parameters[0].min(), parameters[0].max(), parameters[1].min(), parameters[1].max()]
    plt.imshow(pspace.T, cmap='cubehelix', origin = 'bottom', extent=extent, aspect = 'auto')
    plt.ylabel('linewidth hwhm: dx')
    plt.xlabel('resonance position: x0')
    plt.title('all parameters, setting: x = .6')
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    self_test()
