__author__ = 'Bob McMichael'

import numpy.ma as ma
import numpy as np


class ProbDistFunc:
    """
    A probability distribution function class

    Attributes:
        PDF (ndarray): an array representing a probability distribution function.  PDF will be
            multidimensional for joint distributions, one axis for each parameter.
        lnPDF (ndarray): The :code:`log` of :code:`PDF`.
        paramvals (tuple of arrays): a tuple of 1D arrays, each containing discrete values of the
            corresponding parameter.
        shape (tuple): the lengths of the arrays in :code:`paramvals`.
        dims (int): the number of parameters.

    """

    def __init__(self):
        self.Ndraws = 200

    def pdf_config(self, paramvals):
        """initializes the probability distribution function

        Sets up an N-dimensional numpy array as a discrete representation of a joint probability.
        Assumes that the probability is zero outside the parameter space defined by
        :code:`paramvals`.

        Args:
            paramvals (tuple of lists, required): Each list (or array) in paramvals contains
                allowed values of a parameter. Individual lists may represent values of a naturally
                discrete parameter, or they may be discrete approximations of continuous
                parameters.
        """
        # paramvals is expected to be a tuple of arrays containing all possible parameter values
        self.paramvals = paramvals
        # determine the shape of the PDF required to accommodate all of the paramvals
        shape = ()
        for paramarray in paramvals:
            shape += (len(paramarray),)
        self.pdfshape = shape
        self.pdfdims = len(self.pdfshape)
        # create arrays to hold the lnPDF and PDF
        # initialized to a flat probability distribution
        self.lnPDF = np.zeros(self.pdfshape)
        self.PDF = np.ones(self.pdfshape)

    def set_pdf(self, flat=False, probvalarrays=None, pdf=None, lnpdf=None):
        """Initializes the probability distribution function

        Initializes the distribution using the first non-default argument.  Normalization is not
        enforced.  Initializes both :code:`PDF` and :code:`lnPDF`.

        Args:
            flat (bool): If :code:`True`, sets :code:`PDF` values to one and :code:`lnPDF`
                values to zero.  Defaults to :code:`False`.
            probvalarrays (tuple of arrays): Initializes :code:`PDF` to the outer product of the
                input arrays.  Dimensions of probvalarrays members must match dimensions of
                paramvals arrays. Defaults to :code:`None`.
            pdf (list or ndarray):  Specifies the values of :code:`PDF` directly.
                Defaults to :code:`None`.
            lnpdf (list or ndarray): Specifies the values of :code:`lnPDF` directly.
                Defaults to :code:`None`.
        """

        if flat:
            self.lnPDF = np.zeros(self.pdfshape)
            self.PDF = np.ones(self.pdfshape)
            return
        if probvalarrays is not None:
            # Used when the user initializes with a 1-D array of probabilities for each variable
            # parameter
            self.PDF = self.__multiply_probs(probvalarrays)
            self.lnPDF = np.log(self.PDF)
            return
        if pdf is not None:
            if np.array(pdf).shape != self.pdfshape:
                pass  # TODO: raise an error - pdf shape doesn't fit parameter values
            else:
                self.PDF = np.array(pdf)
                self.lnPDF = np.log(pdf)
            return
        if lnpdf is not None:
            if np.array(lnpdf).shape != self.pdfshape:
                pass  # TODO: raise an error - pdf shape doesn't fit parameter values
            else:
                self.lnPDF = np.array(lnpdf)
                self.PDF = np.exp(lnpdf)
            return

    def add_lnpdf(self, lnlikelihood):
        """Adds the argument to the log probability

        A method of multiplying :code:`self.PDF` by another probability distribution by adding
        the logarithms.  Useful in calculating Bayes rule, with a *likelihood* when its logarithm
        is available.  Updates both :code:`PDF` and :code:`lnPDF`.  Normalization is not
        enforced.  See also :code:`multiply_pdf()`.

        Args:
            lnlikelihood: An array representing the :code:`log` of a probability distribution.
        """
        # add the log of likelihood of a measurement to update the PDF
        self.lnPDF += lnlikelihood
        # pseudo-normalize to max lnPDF=0 --> max PDF = 1
        self.lnPDF -= self.lnPDF.max()
        self.PDF = np.exp(self.lnPDF)

    def multiply_pdf(self, likelihood):
        """Multiplies the probability distribution by the argument.

        Useful in calculating Bayes rule, with a *likelihood*.  Updates both :code:`PDF` and
        :code:`lnPDF`.  Normalization is not enforced. See also :code:`add_lnpdf()`.

        Args:
             likelihood: An array representing probability distribution.
        """
        self.lnPDF += np.log(likelihood)
        # pseudo-normalize to max lnPDF=0 --> max PDF = 1
        self.lnPDF -= self.lnPDF.max()
        self.PDF = np.exp(self.lnPDF)

    def markov_draws(self):
        """Takes random draws from the probability distribution.

        Produces :code:`self.Ndraws` samples from the probability distribution function
        using a Markov chain process.

        Returns:
            numpy array: An array containing :code:`Ndraws` sets of parameters.

        """
        return self.__markov_chain_gen_n_dims(self.Ndraws, self.Ndraws)

    def max_params(self):
        """Finds the maximum probability.

        Returns:
            tuple of floats: Parameter values corresponding to the maximum of the probability
            distribution.

        """
        # argmax returns the index of the maximum of the flattened version of self.PDF
        maxindex = np.argmax(self.PDF)
        # calculate indices of the unflattened version
        ix = np.unravel_index(maxindex, self.pdfshape)
        # list comprehension to get the parameter values corresponding to the indices
        maxpars = [p[i] for p, i in zip(self.paramvals, ix)]

        # convert to tuple because our models expect tuples.
        return tuple(maxpars)

    def get_pdf(self, denuisance=(), normalize=True):
        """Normalizes the probability distribution and integrates out nuisance parameters.

        Args:
            denuisance (tuple, optional): Parameters to be integrated out of self.PDF.
                Defaults to :code:`None`.
            normalize (bool, optional): If :code:`True`, normalize sum=1 (default yes)

        Returns:
            numpy array: The probability distribution
        """
        return self.__normalize(self.__denuisance_pdf(denuisance), normalize)

    def get_std(self, paraxis):
        """Computes the standard deviation of the distribution along the specified axis.

        Sums the probability distribution over all axes *except* parax, producing a 1D
        distribution, then calculates the standard deviation of that 1D distribution.

        Args:
            paraxis (int, required): The axis index corresponding to the parameter of interest.

        Returns:
            float: The standard deviation along the :code:`paraxis` axis.
        """
        # We need a list of axes to sum, i.e. all axes _except_ for the requested paraxis,
        # all axes
        axes = list(np.arange(self.pdfdims))
        # remove the parameter axis
        del axes[paraxis]

        # get the collapsed pdf
        one_d_pdf = self.get_pdf(denuisance=axes, normalize=True)
        # and the corresponding parameter values
        one_param = np.array(self.paramvals[paraxis])

        # calculate the standard deviation using sums
        pbar = np.sum(one_d_pdf * one_param)
        psquare = np.sum(one_d_pdf * one_param ** 2)
        ssquare = psquare - pbar ** 2
        return np.sqrt(ssquare)

    def get_mean(self, paraxis):
        """Computes the mean and standard deviation of the distribution along the specified axis.

        Sums the probability distribution over all axes *except* parax, producing a 1D
        distribution, then calculates the mean standard deviation of that 1D distribution.

        Args:
            paraxis (int, required): The axis index corresponding to the parameter of interest.

        Returns:
            tuple of floats: The mean and standard deviation along the :code:`paraxis` axis.
        """
        # We need a list of axes to sum, i.e. all axes _except_ for the requested paraxis,
        # all axes
        axes = list(np.arange(self.pdfdims))
        # remove the parameter axis
        del axes[paraxis]

        # get the collapsed pdf
        one_d_pdf = self.get_pdf(denuisance=axes, normalize=True)
        # and the corresponding parameter values
        one_param = np.array(self.paramvals[paraxis])

        # calculate the standard deviation using sums
        pbar = np.sum(one_d_pdf * one_param)
        psquare = np.sum(one_d_pdf * one_param ** 2)
        ssquare = psquare - pbar ** 2
        return pbar, np.sqrt(ssquare)

    def entropy(self):
        """Evaluates the information entropy of the distribution.

        Information entropy is a measure of the "sharpness" of the distribution, largest for a
        uniform distribution and smallest when the probability is nonzero only in one array
        element.

        Returns:
            float: The information entropy.
        """
        return self.__calculate_discrete_entropy(self.PDF)

    ##########################################################
    # The nitty-gritty details, in private methods
    ##########################################################

    def __denuisance_pdf(self, denuisance=()):
        """
        take the PDF and sum over axes indicated by the nuisance tuple
        then reshape
        :param denuisance:  a tuple indicating axes to be 'integrated' out
        :return: nparray with collapsed, de-nuisanceified pdf
        """
        if denuisance:
            return np.sum(self.PDF, axis=tuple(denuisance))
        else:
            # If denuisance = False (default), skip the procedure and return PDF
            return self.PDF

    def __normalize(self, a_pdf, normalize=True):
        """
        Normalize the input array
        :param a_pdf:
        :return:
        """
        if normalize:
            return a_pdf / a_pdf.sum()
        else:
            return a_pdf

    def __multiply_probs(self, probvals):
        # Used when the user initializes with a 1-D array of probabilities for each variable
        # parameter

        # sift out just the variables
        meshes = list(np.meshgrid(*probvals, indexing='ij'))

        # probability arrays must have same shape and order as parameter arrays
        if meshes[0].shape != self.pdfshape:
            pass  # raise an error -

        # multiply all of the probabilities together
        pdf = np.ones(self.pdfshape)
        for mesh in meshes:
            pdf *= mesh
        return pdf

    def __randdraw(self, r, dist):
        """
        Draw a random index value from a 1-D distribution
        :param r: a random number btw 0:1
        :param dist:  a distribution "function" as a 1-D array, maybe unnormalized
        :return: randomly drawn index from dist
        """
        # the strategy is to compare a random number with the running integral (cumulative sum)
        # of the pdf
        cumdist = np.cumsum(dist)
        # rather than normalize the whole distribution, we'll just scale up the random numbers so
        #  that they span [0:cumdist[-1]).  the last value of cumdist is the "integral" of the dist.
        rscaled = r * cumdist[-1]

        # Next, find the index where cumdist > rscaled
        bigger = np.array(np.nonzero(cumdist > rscaled))
        # nonzero returns a tuple with the array of indices in the first element
        if bigger.shape[1] == 0:
            return len(cumdist) - 1
        else:
            return bigger[0][0]

    def __markov_chain_gen_n_dims(self, n_draws, n_burn=100):
        """
        produce a number of samples from a probability distribution function
        using a Markov chain process
        :param n_draws:     the number of samples to generate
        :param n_burn:    desired number of "pre-randomizing" moves
        :return:
        """

        # Getting to know youuuu, getting to know all about youuuu!
        # learn about the pdf
        ndims = len(self.pdfshape)

        # generate a random location to start
        location = []
        for i, paramarray in enumerate(self.paramvals):
            # append a random position along an axis, but only for variable parameters
            location.append(np.random.randint(len(paramarray)))

        # while True:
        # The initial burn-in: n_burn full Markov moves
        # A full move here is one where the position in the pdf changes in all directions\
        # first, a collection of random numbers to draw from
        rand_bucket = np.random.random((n_burn, ndims))

        # making n_burn full moves
        for i in np.arange(n_burn):
            # making one step along each axis
            for axis in np.arange(ndims):
                # substitute an ellipsis as an index in the location to get the probabilities
                # along one axis -- it's a python thing.
                location[axis] = ...
                # pdf[location] will now return a 1D array that we interpret as a pdf,
                # and we make a random draw for the new location along the axis
                location[axis] = self.__randdraw(rand_bucket[i, axis], self.PDF[location])

        # Now that the Markov chain has been initialized, we move to generating the actual draws
        # from the pdf each full move generates Ndims draws, so we'll need this many moves.
        nmoves = int(n_draws / ndims) + 1
        # a stash of random numbers
        rand_pail = np.random.random((nmoves, ndims))
        # a place for the draws to go, perhaps slightly more spaces than needed for n_draws

        draws = np.zeros((nmoves * ndims, ndims), dtype='int')

        i_draw = 0  # a counter
        # make the full moves
        for i in np.arange(nmoves):
            # make one step along each axis as before
            for axis in np.arange(ndims):
                location[axis] = ...
                location[axis] = self.__randdraw(rand_pail[i, axis], self.PDF[location])
                # store the new location
                draws[i_draw, :] = np.array(location)
                i_draw += 1

        # all done with our random walk
        # so now we have a bunch of draws in terms of index location in the PDF array.
        # But, we really need the actual parameter values
        paramdraws = np.zeros((n_draws, ndims))
        # stepping through the required number of draws
        for i, location in enumerate(draws[:n_draws]):
            # store a the parameter value
            for j, param in enumerate(self.paramvals):
                paramdraws[i, j] = param[location[j]]

        return paramdraws

    def __calculate_discrete_entropy(self, a_pdf):
        """
        Calculate the entropy of a discrete distribution.
        :param a_pdf: an array of probability density possibly unnormalized
        :return: sum of x * log2(x)
        """
        # "There's an app for that" --> scipy.stats.entropy does almost the same thing, but without
        # flattening and also offering the Kullback-Leibler divergence and offering different log
        # bases.
        #   First flatten the distro array to 1D - we're just going to add things up anyways
        # Clean out the zeros (anything less than 1e-100) because zeros cause problems and
        # contribute nothing to either normalization or entropy integrals
        maskeddist = ma.masked_less(a_pdf.flatten(), 1e-100).compressed()
        # normalization
        norm = maskeddist.sum()
        # the obvious thing would be to divide the whole maskeddist array by norm.
        # However, we can avoid an array operation by handling the norm separately
        #   Sum( P/n * ln (P/n) )
        # = Sum( P*ln(P) )/n - Sum(P)*ln(n)/n
        # = Sum( P*ln(P) )/n - ln(n)
        # entropy integrand
        integrand = maskeddist * np.log2(maskeddist)
        return -integrand.sum() / norm + np.log2(norm)
