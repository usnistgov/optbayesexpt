__author__ = 'Bob McMichael'

import numpy as np
from .exptmodel import ExptModel
from .probdistfunc import ProbDistFunc


class OptBayesExpt(ExptModel, ProbDistFunc):
    """An implementation of Optimal Bayesian Experimental Design

    OptBayesExpt provides an efficient alternative to the traditional measure-then-fit approach
    to measurement.  Instead of fitting parameters of a model function to a existing data,
    optimal Bayesian experimental design incorporates the model into the measurement process,
    using it to determine an informed measurement strategy during the measurement process.

    The method calculates a *utility function* of measurement settings, which is qualitatively a
    measure of the predicted improvement to the parameter distribution.  See the manual for more
    detail.

    Important:

        The :obj:`ExptModel.model_function()` method must be redefined to incorporate a model
        that's relevant to the application.

    Attributes:
        sets (:obj:`tuple` of :obj:`ndarray`): Each array in the :code:`sets` tuple contains the
            possible discrete values of a measurement setting.  Applied voltage, excitation frequency,
            and a knob that goes to eleven are all examples of settings.  For computational
            speed, it is important to keep setting arrays as few and small as practical.
            Settings arrays that cover unused setting values, or that use overly fine
            disretization will slow the calculations.  Settings that are held constant belong in
            the :code:`cons` array.
        pars (:obj:`tuple` of :obj:`ndarray`):  Each array in the :code:`pars` tuple contains the
            possible values of a model parameter.  In a simple example model, :code:`y = m * x +
            b`, the parameters are :code:`m` and :code:`b`.  As with the :code:`sets`,
            :code:`pars` arrays should be kept few and small.  Parameters that can be assumed
            constant belong in the :code:`cons` array.  Discretization should only be fine enough
            to support the needed measurement precision.  The parameter ranges must also be
            limited: too broad, and the computation will be slow; too narrow, and the measurement
            may have to be adjusted and repeated.
        cons (:obj:`tuple` of :obj:`float`):  Model constants.  Examples include experimental
            settings that are rarely changed, and model parameters that are well-known from previous
            measurement results.

    """

    def __init__(self):
        self.sets = ()
        self.pars = ()
        self.cons = ()

        ExptModel.__init__(self)
        # provides
        #  self.allsettings
        #  self.allparams
        #  self.constants
        # and methods for evaluating the model function
        #  self.model_function  (empty!)
        #  self.eval_over_all_parameters
        #  self.eval_over_all_settings

        ProbDistFunc.__init__(self)
        # provides
        #  self.paramvals
        #  self.pdfshape
        #  self.PDF
        #  self.lnPDF
        # and methods
        #  set_pdf
        #  add_lnpdf
        #  multiply_pdf
        #  markov_draws

    def config(self):
        """
        Configure the experimental model and the parameter pdf.

        This method should be called before either pdf_update or opt_setting
        """
        self.model_config(self.sets, self.pars, self.cons)
        self.pdf_config(self.pars)

    def pdf_update(self, onesetting, ymeas, sigma=1):
        """
        Refines the parameters' probability distribution function given a measurement result.

        An implementation of Bayesian inference, uses the model to calculate the likelihood of
        obtaining the measurement result :code:`ymeas` as a function of parameter values,
        and uses that likelihood to generate a refined *posterior* (after-measurement) distribution
        from the *prior* (pre-measurement) distribution.

        Args:
            onesetting (:obj:`tuple` of :obj:`float`): The measurement settings.
            ymeas: (:obj:`float`): The measurement mean value.
            sigma: (:obj:`float`): The uncertainty of the measurement expressed as a standard
            deviation.

        """

        # calculate the model for all values of the parameters
        ymodel = self.eval_over_all_parameters(onesetting)

        # Assuming the measurement is drawn from a Gaussian(sigma) distribution of possible
        # measurement results, the likelihood is the probability of getting that particular
        # result given any parameter combination
        lnlikelihood = -((ymodel - ymeas) / sigma)**2 / 2

        # uptdate the pdf
        # multiply the pdf by the likelihood or add the lnlikelihood
        self.add_lnpdf(lnlikelihood)

    def calc_exp_utility(self):
        """
        Estimate the exponential of utility as a function of settings

        Used in selecting measurement settings. For each setting combination, calculate the
        standard deviation of model outputs produced
        by modeling a set of random draws from the parameter distribution.  Loosely, the spread
        in model outputs given the distribution of parameters.  In the theory, the *utility* is
        the change in the information entropy of the distribution, which involves a :code:`log(
        )`.  This function skips the :code:`log()`, and calculates :code:`exp` (*utility*).

        Returns:
            exp_utility as an :obj:`ndarray` with dimensions of :code:`allsettings`
        """
        # get parameter sets drawn from the pdf for a sampling of model outputs
        paramsets = self.markov_draws()

        # make space for model results
        ycalc = np.zeros((self.Ndraws, ) + self.allsettings[0].shape)
        # the default for the default number of draws is set in ProbDistFunc__init__()
        # self.allsettings  is a tuple of meshgrid output arrays.  One is enough to determine the
        #  shape of setting space.

        # fill the model results for each drawn parameter set
        for i, oneparamset in enumerate(paramsets):
            ycalc[i] = self.eval_over_all_settings(oneparamset)

        # Evaluate how much the model varies at each setting
        # calculate the std for each setting
        exp_utility = np.std(ycalc, axis=0)

        return exp_utility

    def opt_setting(self):
        """Calculate a setting with the best probability of refining the pdf

        At what settings are we most uncertain about how an experiment will come out? That is
        where the next measurement will do the most good. So, we calculate model outputs
        for a bunch of possible model parameters and see wherethe output varies the most.
        We use our accumulated knowledge by drawing the possible parameters from the current
        parameter pdf.

        Returns:  A tuple of settings.
        """

        exp_utility = self.calc_exp_utility()

        # Find the settings with the maximum utility
        # argmax returns an array of indices into the flattened array
        bestindices = np.argmax(exp_utility)

        # translate to setting values
        # allsettings is a list of setting arrays generated by np.meshgrid, one for each 'knob'
        bestvalues = [setting.flatten()[bestindices] for setting in self.allsettings]
        # list comprehension - woohoo!
        return tuple(bestvalues)

    def good_setting(self, pickiness=1):
        """
        Calculate a setting with a good probability of refining the pdf

        In comparison to the opt_setting method, where the measurements concentrate on the
        settings,
        leaving much of setting space untouched.   How do you know you're not missing something
        in that region?  By allowing some randomness in the setting selection, we allow
        non-optimum settings, covering setting space lightly while still emphasizing the
        settings that improve our parameter estimates.

        Args:
           pickiness (float): A setting selection tuning parameter.  Pickiness=0 produces random
              settingss.  With pickiness values greater than about 10 the behavior is similar to
              :code:`opt_setting()`.

        Returns:
            A tuple of settings for the next measurement
        """

        utility = self.calc_exp_utility() ** pickiness

        # the exponent 'pickiness' is a tuning parameter

        # yvar array should have dimensions of allsettings
        # Now, we're going to treat yvar as a distribution and select a value
        # cumsum flattens ystddev to 1-D
        # We'll have to use np.unravel_index() to unflatten it
        cumsum_eu = np.cumsum(utility)
        # a random number, scaled to account for the un-normalized "distribution"
        rscaled = np.random.rand()*cumsum_eu[-1]

        # Next, find the index where cumsumYdev > rscaled
        bigger = np.array(np.nonzero(cumsum_eu > rscaled))
        # nonzero returns a tuple with the array of indices in the first element
        if bigger.shape[1] == 0:
            goodindex = len(cumsum_eu) - 1
            print('overflow')
        else:
            goodindex = bigger[0][0]

        # find the corresponding indices of the unflattened settings arrays
        goodindices = np.unravel_index(goodindex, utility.shape)

        # translate to setting values
        # allsettings is a tuple of setting arrays generated by np.meshgrid
        goodvalues = [setting[goodindices] for setting in self.allsettings]
        # list comprehension - woohoo!
        return tuple(goodvalues)

    def besty(self):
        """Evaluate the model function at the maximum-probability parameters

        Similar in spirit to the "best-fit curve", the :code:`model_function` is evaluated over
        all settings value using the maximum-probability parameters.

        Returns:
            An array of :code:`model_function` outputs with dimensions
            of :code:`ExptModel.allsettings`.
        """
        bestparams = self.max_params()
        return self.eval_over_all_settings(bestparams)

