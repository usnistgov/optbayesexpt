__author__ = 'Bob McMichael'

import numpy.ma as ma
import numpy as np


class ProbDistFunc:
    """
    A probability distribution function class
    variables:
        lnPDF -- an N-dimensional array, one axis for each variable parameter
            contains nat.log of the probability, not normalized
        shape -- tuple describing dimensions of pdf
        paramspace -- a tuple of arrays and constants intended for iterating over all of parameter space
        paramisarray -- which parameters are arrays, i.e. variables
    methods:
        __init__ -- determines the pdf's shape from lists/arrays of possible parameter values
             -- creates paramspace for model evaluation over all parameter space
             -- determines which parameters are variabls
        set_pdf -- initial value for the pdf
        update_pdf --  updates the lnPDF by (typically) adding a lnlikelyhood
        markov_draws -- generates an array of parameter sets using a random walk in the pdf.
        entropy  -- calculate the entropy of the pdf
    """

    def __init__(self):
        self.Ndraws=200

    def pdf_config(self, paramvals):
        # paramvals is expected to be a tuple of arrays containing all possible parameter values
        # nuisance is
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

    def set_pdf(self, flat=False, probvalarrays=[], pdf=[], lnpdf=[]):
        # set the pdf with some initial guess or restore a saved version
        done = 0
        if flat:
            self.lnPDF = np.zeros(self.pdfshape)
            self.PDF = np.ones(self.pdfshape)
            done += 1
        if probvalarrays:
            # Used when the user initializes with a 1-D array of probabilities for each variable parameter
            self.PDF = self.__multiply_probs(probvalarrays)
            self.lnPDF = np.log(self.PDF)
            done += 1
        if pdf:
            if np.array(pdf).shape != self.pdfshape:
                pass  # raise an error - pdf shape doesn't fit parameter values
            else:
                self.PDF = pdf
                self.lnPDF = np.log(pdf)
                done += 1
        if lnpdf:
            if np.array(pdf).shape != self.pdfshape:
                pass  # raise an error - pdf shape doesn't fit parameter values
            else:
                self.lnPDF = lnpdf
                self.PDF = np.exp(lnpdf)
                done += 1
        if done != 1:  # the pdf should be set exactly once
            pass  # raise an error

    def add_lnpdf(self, lnlikelihood):
        # add the log of likelihood of a measurement to update the PDF
        self.lnPDF += lnlikelihood
        # pseudo-normalize to max lnPDF=0 --> max PDF = 1
        self.lnPDF -= self.lnPDF.max()
        self.PDF = np.exp(self.lnPDF)

    def multiply_pdf(self, likelihood):
        # multiply by another
        self.lnPDF += np.ln(likelihood)
        # pseudo-normalize to max lnPDF=0 --> max PDF = 1
        self.lnPDF -= self.lnPDF.max()
        self.PDF = np.exp(self.lnPDF)

    def markov_draws(self):
        """
        produce a number of samples from a probability distribution function
        using a Markov chain process
        :param n_draws:   the number of samples to generate
        :param n_burn:    desired number of "pre-randomizing" moves
        :return: a list of parameter combinations
        """

        return self.__markov_chain_gen_ND(self.Ndraws, self.Ndraws)

    def max_params(self):
        """
        find the parameters corresponding to the max probability
        :return:
        """
        # argmax returns the index of the maximum of the flattened version of self.PDF
        maxindex = np.argmax(self.PDF)
        # calculate indices of the unflattened version
        ix = np.unravel_index(maxindex, self.pdfshape)
        # list comprehension to get the parameter values corresponding to the indices
        maxpars = [p[i] for p, i in zip(self.paramvals, ix)]
        # convert to tuple because our models expect tuples.
        print(maxpars)
        return tuple(maxpars)

    def get_PDF(self, denuisance=(), normalize=True):
        """
        packaging & polishing the PDF
        Integrate over nuisance parameters, normalize and return the resulting pdf.
        :param denuisance:   tuple identifying parameters to be integrated out of self.PDF (default none)
        :param normalize:  Boolean to normalize sum=1 (default yes)
        :return:
        """
        return self.__normalize(self.__denuisance_pdf(denuisance), normalize)

    def get_std(self, paraxis):
        """
        compute the standard deviation of the PDF collapsed down to 1 axis
        :param paraxis: the axis of the pdf corresponding to the vaiable of interest.
        :return: standard deviation  (float)
        """
        # We need a list of axes to sum, i.e. all axes _except_ for the requested paraxis,
        # all axes
        axes = list(np.arange(self.pdfdims))
        # remove the parameter axis
        del axes[paraxis]

        # get the collapsed pdf
        oneDpdf = self.get_PDF(denuisance=axes, normalize=True)
        # and the corresponding parameter values
        oneParam = np.array(self.paramvals[paraxis])

        # calculate the standard deviation using sums
        pbar = np.sum(oneDpdf * oneParam)
        psquare = np.sum(oneDpdf * oneParam **2)
        ssquare = psquare - pbar**2
        return np.sqrt(ssquare)

    def entropy(self):
        return self.__calculate_discrete_Entropy(self.PDF)

    """The nitty-gritty details, in private methods"""

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
        # Used when the user initializes with a 1-D array of probabilities for each variable parameter
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
        # the strategy is to compare a random number with the running integral (cumulative sum) of the pdf
        cumdist = np.cumsum(dist)
        # rather than normalize the whole distribution, we'll just scale up the random numbers
        # so that they span [0:cumdist[-1]).  the last value of cumdist is the "integral" of the dist.
        rscaled = r * cumdist[-1]

        # Next, find the index where cumdist > rscaled
        bigger = np.array(np.nonzero(cumdist > rscaled))
        # nonzero returns a tuple with the array of indices in the first element
        if bigger.shape[1] == 0:
            return len(cumdist) - 1
        else:
            return bigger[0][0]

    def __markov_chain_gen_ND(self, n_draws, n_burn=100):
        """
        produce a number of samples from a probability distribution function
        using a Markov chain process
        :param pdf:         an N-dimensional pdf
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
                # substitute an ellipsis as an index in the location to get the probabilities along one axis
                # it's a python thing.
                location[axis] = ...
                # pdf[location] will now return a 1D array that we interpret as a pdf, and we make a random draw
                # for the new location along the axis
                location[axis] = self.__randdraw(rand_bucket[i, axis], self.PDF[location])

        # Now that the Markov chain has been initialized, we move to generating the actual draws from the pdf
        # each full move generates Ndims draws, so we'll need this many moves.
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

    def __calculate_discrete_Entropy(self, a_pdf):
        """
        Calculate the entropy of a discrete distribution.
        :param distro: an array of probability density possibly unnormalized
        :return: sum of x * log2(x)
        """
        """
        "There's an app for that" --> scipy.stats.entropy does almost the same thing, but without flattening
        and also offering the Kullback-Leibler divergence and offering different log bases.
        """
        # flatten the distro array to 1D - we're just going to add things up anyways
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


def self_test():
    xmin, xmax, ymin, ymax = (-1, 1, -2, 2)
    extent = [xmin, xmax, ymin, ymax]

    xparam = np.linspace(xmin, xmax, 101)
    yparam = np.linspace(ymin, ymax, 201)

    #mypdf = ProbDistFunc((xparam, yparam))
    mypdf = ProbDistFunc()
    mypdf.pdf_config((xparam,yparam))

    print('default mypdf.shape = {}'.format(mypdf.pdfshape))
    print('default mypdf.lnPDF = {}'.format(mypdf.lnPDF))
    print('default mypdf.PDF = {}'.format(mypdf.PDF))

    import matplotlib.pyplot as plt

    plt.figure()

    plt.subplot(221)
    plt.title('default')
    plt.imshow(mypdf.PDF.T, extent=extent, cmap='cubehelix', aspect='auto')
    plt.colorbar()

    plt.subplot(222)
    plt.title('probvalarrays')
    xprobs = 1 - xparam * xparam / 1.1
    yprobs = 1 - yparam * yparam / 8
    mypdf.set_pdf(probvalarrays=(xprobs, yprobs))
    plt.imshow(mypdf.PDF.T, extent=extent, cmap='cubehelix', aspect='auto')
    plt.colorbar()

    plt.subplot(223)
    plt.title('exp')
    XX, YY = np.meshgrid(xparam, yparam, indexing='ij')
    print('XX.shape = {}'.format(XX.shape))
    mypdf.set_pdf(pdf=np.exp(-(XX * XX + YY * YY + XX * YY) * 5))
    plt.imshow(mypdf.PDF.T, origin='bottom', extent=extent, cmap='cubehelix', aspect='auto')
    plt.colorbar()

    plt.subplot(224)
    plt.title('draws')
    # mydraws = mypdf.markov_draws(500)
    mydraws = mypdf.markov_draws()

    plt.scatter(mydraws[:, 0], mydraws[:, 1], marker='.')
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    self_test()
