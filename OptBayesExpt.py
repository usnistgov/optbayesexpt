__author__ = 'Bob McMichael'

import numpy as np
from ExptModel_class import ExptModel
from ProbDistFunc_class import ProbDistFunc

class OptBayesExpt(ExptModel, ProbDistFunc):
    """
    combines an ExptModel class and a ProbDistFunc class to steer an experiment
    ExptModel provides methods for evaluating an experimental model
        * Note that the actual model ins't provided.
        * To add the model, make a child class of BayesOptExpt and redefine the model_function method.
    ProbDistFunc provides a probability distribution function and methods for parameter probabilities
    Add-ons provided by BayesOptExpt include
        * a method to update the pdf based on a new measurement
        * a method to estimate settings for the next measurement
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
        :param settings:   list of arrays or list of floats -- experimental settings
        :param parameters: list of floats or list of arrays -- model parameters
        :param constants:  list of floats  -- constants in the model
        :return:
        """
        self.model_config(self.sets, self.pars, self.cons)
        self.pdf_config(self.pars)

    # new functionality
    # recieve a new measurement value and uncertainty (ymeas, sigma) make at a setting (onesetting) and
    # update the pdf to incorporate the new information.

    def pdf_update(self, onesetting, ymeas, sigma=1):
        """
        Input a new measurement value and uncertainty (ymeas, sigma) make at a setting (onesetting) and
        update the pdf to incorporate the new information.
        Calculate the log(likelihood) for a result ymeas from a measurement made at onesetting
        And then update the pdf
        :param onesetting: Measurement settings
        :param ymeas:      Measurement result
        :param sigma:      Measurement std dev
        :return:
        """
        # calculate the model for all values of the parameters
        ymodel = self.eval_over_all_parameters(onesetting)
        # Assuming the measurement is drawn from a Gaussian(sigma) distribution of possible measurement results,
        # the likelihood is the probability of getting that particular result given any parameter combination
        lnlikelihood = -((ymodel - ymeas) / sigma)**2 / 2
        # uptdate the pdf
        # multiply the pdf by the likelihood or add the lnlikelihood
        self.add_lnpdf(lnlikelihood)

    def opt_setting(self):
        """
        Calculate a setting with the best probability of refining the pdf
        At what settings are we most uncertain about how an experiment will come out?
        That is where the next measurement will do the most good.
        So, we calculate model outputs for a bunch of possible model parameters and see where the output varies the most.
        We use our accumulated knowledge by drawing the possible parameters from the current parameter pdf.
        :param Nsamples:
        :param Nburn:
        :return:
        """
        # get parameter sets drawn from the pdf for a sampling of model outputs
        paramsets = self.markov_draws()

        # make space for model results
        ycalc = np.zeros((self.Ndraws, ) + self.allsettings[0].shape)
        # the default for the default number of draws is set in ProbDistFunc__init__()
        # self.allsettings is a tuple of meshgrid output arrays.  One is enough to determine the shape of
        # setting space.

        # fill the model results for each drawn parameter set
        for i, oneparamset in enumerate(paramsets):
            ycalc[i] = self.eval_over_all_settings(oneparamset)

        # Evaluate how much the model varies at each setting
        # The hard-core theory uses information entropy as the measure.
        # Entropy counts probabilities that are neither 0 or 1  Sum[p(x) ln(p(x)], but doesn't capture
        # how those probability values are arranged
        # Std deviation is faster to calculate and reflects the 'badness' of bimodal distributions
        # Translation:  Faster.  Runs good.

        # calculate the std for each setting
        ystddev = np.std(ycalc, axis = 0)

        # find the largest std dev.
        # argmax returns an array of indices
        bestindices = np.argmax(ystddev)

        # translate to setting values
        # allsettings is a list of setting arrays generated by np.meshgrid
        bestvalues = [setting[bestindices] for setting in self.allsettings]
        # list comprehension - woohoo!
        return tuple(bestvalues)

    def good_setting(self, pickiness=1):
        """
        Calculate a setting with a good probability of refining the pdf, but not necessarily the best.
        In opt_setting method, the measurements often concentrate on a few settings, leaving much of setting space
        untouched.   How do you know you're not missing something in that region?  By allowing some randomness in the
        setting selection, we allow non-optimum settings, covering setting space lightly while still emphasizing the
        settings that improve our parameter estimates.
        :param pickiness  - a setting selection tuning parameter.  Pickiness=0 produces random settingss.  Large
                pickiness values emphasize the optimum settings.
        :return:  a tuple of settings for the next measurement
        """
        # get parameter sets drawn from the pdf for a sampling of model outputs
        paramsets = self.markov_draws()

        # make space for model results
        ycalc = np.zeros((self.Ndraws,) + self.allsettings[0].shape)
        # the default for the default number of draws is set in ProbDistFunc__init__()

        # fill the model results for each drawn parameter set
        for i, oneparamset in enumerate(paramsets):
            ycalc[i] = self.eval_over_all_settings(oneparamset)

        # Evaluate how much the model varies at each setting
        # The hard-core theory uses information entropy as the measure.
        # Entropy counts probabilities that are neither 0 or 1  Sum[p(x) ln(p(x)], but doesn't capture
        # how those probability values are arranged
        # Variance is faster to calculate and reflects the 'badness' of bimodal distributions
        # Translation:  Faster.  Runs good.

        # calculate the variance for each setting
        yvar = np.var(ycalc, axis=0)**pickiness
        # the exponent 'pickiness' is a tuning parameter

        # yvar array should have dimensions of allsettings
        # Now, we're going to treat yvar as a distribution and select a value
        # cumsum flattens ystddev to 1-D
        # We'll have to use np.unravel_index() to unflatten it
        cumsumYvar = np.cumsum(yvar)
        # a random number, scaled to account for the un-normalized "distribution"
        rscaled = np.random.rand()*cumsumYvar[-1]

        # Next, find the index where cumsumYdev > rscaled
        bigger = np.array(np.nonzero(cumsumYvar > rscaled))
        # nonzero returns a tuple with the array of indices in the first element
        if bigger.shape[1] == 0:
            goodindex = len(cumsumYvar) - 1
            print('overflow')
        else:
            goodindex = bigger[0][0]

        # find the corresponding indices of the unflattened settings arrays
        goodindices = np.unravel_index(goodindex, yvar.shape)

        # translate to setting values
        # allsettings is a tuple of setting arrays generated by np.meshgrid
        goodvalues = [setting[goodindices] for setting in self.allsettings]
        # list comprehension - woohoo!
        return tuple(goodvalues)

    def besty(self):
        bestparams = self.max_params()
        return self.eval_over_all_settings(bestparams)

    #a few methods to manage these input tuples

    # clearing data
    def clrsets(self):
        self.sets = ()

    def clrpars(self):
        self.pars = ()

    def clrcons(self):
        self.cons = ()

    # appending data to tuples
    def addsets(self, setarray):
        self.sets += (setarray, )

    def addpars(self, pararray):
        self.pars += (pararray, )

    def addcon(self, constval):
        self.cons += (constval, )

    # reporting data
    def getsets(self):
        return self.sets

    def getpars(self):
        return self.pars

    def setcons(self):
        return self.cons


if __name__ == '__main__':
    # settings
    s = np.linspace(0, 10, 5)
    svs = (s,)
    p1 = np.linspace(-1, 1, 11)
    p2 = np.linspace(2.1, 3.9, 9)
    pvs = (p1, p2)
    c1 = 1
    c2 = 5
    cvs = (c1, c2)
    myexpt = OptBayesExpt()
    myexpt.sets = svs
    myexpt.pars = pvs
    myexpt.cons = cvs
    myexpt.config()

    print(myexpt.pdfshape)
    print(myexpt.paramvals)
    print(myexpt.allparams)
    print(myexpt.allsettings)