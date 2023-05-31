import numpy as np
import warnings
from optbayesexpt.samplers import sample,Liu_West_resampler

GOT_NUMBA = False
# GOT_NUMBA = True
try:
    from numba import njit, float64
except ImportError:
    GOT_NUMBA = False


class ParticlePDF:
    """A probability distribution function.

    A probability distribution :math:`P(\\theta_0, \\theta_1, \\ldots,
    \\theta_{n\_dims})` over parameter variables :math:`\\theta_i` is
    represented by a large-ish number of samples from the distribution,
    each with a weight value.  The distribution can be visualized as a cloud
    of particles in parameter space, with each particle corresponding to a
    weighted random draw from the distribution.  The methods implemented
    here largely follow the algorithms published in Christopher E Granade et
    al. 2012 *New J. Phys.* **14** 103013.

    Warnings:

        The number of samples (i.e. particles) required for good performance
        will depend on the application.  Too many samples will slow down the
        calculations, but too few samples can produce incorrect results.
        With too few samples, the probability distribution can become overly
        narrow, and it may not include the "true" parameter values. See the
        ``resample()`` method documentation for details.

    Arguments:
        prior (:obj:`2D array-like`):
            The Bayesian *prior*, which initializes the :obj:`ParticlePDF`
            distribution. Each of ``n_dims`` sub-arrays contains
            ``n_particles`` values of a single parameter, so that the *j*\
            _th elements of the sub-arrays determine the coordinates of a
            point in parameter space. Users are encouraged to experiment with
            different ``n_particles`` sizes to assure consistent results.

    Keyword Args:

        resample_threshold (:obj:`float`): Sets a threshold for automatic
            resampling. Resampling is triggered when the effective fraction of
            particles, :math:`1 / (N\\sum_i^N w_i^2)`, is smaller than
            ``resample_threshold``.  Default ``0.5``.

        auto_resample (:obj:`bool`): Determines whether threshold testing and
            resampling are performed when ``bayesian_update()`` is called.
            Default ``True``.

        use_jit (:obj:`bool`): Allows precompilation of some methods for a
            modest increase in speed.  Only effective on systems where
            ``numba`` is installed. Default ``True``

    **Attributes:**
    """

    def __init__(self, prior, resampler=Liu_West_resampler, resample_threshold=0.5,
                 auto_resample=True, use_jit=True, **kwargs):

        #: dict: A package of parameters affecting the resampling algorithm
        #:
        #:     - ``'resample_threshold'`` (:obj:`float`):  Initially,
        #:       the value of the ``resample_threshold`` keyword argument.
        #:       Default ``0.5``.
        #:
        #:     - ``'auto_resample'`` (:obj:`bool`): Initially, the value of the
        #:       ``auto_resample`` keyword argument. Default ``True``.
        self.tuning_parameters = {'resample_threshold': resample_threshold,
                                  'auto_resample': auto_resample}
        self.resampler = resampler
        self.resampler_params = kwargs

        #: ``n_dims x n_particles ndarray`` of ``float64``: Together with
        #: ``particle_weights``,#: these ``n_particles`` points represent
        #: the parameter probability distribution. Initialized by the
        #: ``prior`` argument.
        self.particles = np.asarray(prior)

        #: ``int``: the number of parameter samples representing the
        #: probability distribution. Determined from the trailing dimension
        #: of ``prior``.
        self.n_particles = self.particles.shape[-1]

        #: ``int``: The number of parameters, i.e. the dimensionality of
        #: parameter space. Determined from the leading dimension of ``prior``.
        self.n_dims = self.particles.shape[0]

        #: ``ndarray`` of ``int``: Indices into the particle arrays.
        self._particle_indices = np.arange(self.n_particles, dtype='int')

        #: ndarray of ``float64``: Array of probability weights
        #: corresponding to the particles.
        self.particle_weights = np.ones(self.n_particles) / self.n_particles

        #: ``bool``: A flag set by the ``resample_test()`` function. ``True`` if
        #: the last ``bayesian_update()`` resulted in resampling,
        #: else ``False``.
        self.just_resampled = False

        # Precompile numerically intensive functions for speed
        # and overwrite _normalized_product() method.
        if GOT_NUMBA and use_jit:
            @njit([float64[:](float64[:], float64[:])], cache=True)
            def _proto_normalized_product(wgts, lkl):
                tmp = wgts * lkl
                return tmp / np.sum(tmp)
        else:
            def _proto_normalized_product(wgts, lkl):
                tmp = np.nan_to_num(wgts * lkl)
                result = np.nan_to_num(tmp / np.sum(tmp))
                return result
            self._normalized_product = _proto_normalized_product

        try:
            self.rng = np.random.default_rng()
        except AttributeError:
            self.rng = np.random

    def set_pdf(self, samples, weights=None):
        """Re-initializes the probability distribution

        Also resets ``n_particles`` and ``n_dims`` deduced from the
        dimensions of ``samples``.

        Args:
            samples (array-like):  A representation of the new distribution
                comprising `n_dims` sub-arrays of `n_particles`
                samples of each parameter.
            weights (ndarray): If ``None``, weights will be assigned uniform
                probability.  Otherwise, an array of length ``n_particles``
        """
        self.particles = np.asarray(samples)
        self.n_particles = self.particles.shape[-1]
        self.n_dims = self.particles.shape[0]
        if weights is None:
            self.particle_weights = np.ones(self.n_particles)\
                                    / self.n_particles
        else:
            if len(weights) != self.n_particles:
                raise ValueError('Length of weights does not match the '
                                 'number of particles.')
            else:
                self.particle_weights = weights / np.sum(weights)

    def mean(self):
        """Calculates the mean of the probability distribution.

        The weighted mean of the parameter distribution. See also
        :obj:`std()` and :obj:`covariance()`.

        Returns:
            Size ``n_dims`` array.
        """
        return np.average(self.particles, axis=1,
                          weights=self.particle_weights)

    def covariance(self):
        """Calculates the covariance of the probability distribution.

        Returns:
            The covariance of the parameter distribution as an
            ``n_dims`` X ``n_dims`` array. See also :obj:`mean()` and
            :obj:`std()`.
        """

        raw_covariance = np.cov(self.particles, aweights=self.particle_weights)
        if self.n_dims == 1:
            return raw_covariance.reshape((1, 1))
        else:
            return raw_covariance

    def std(self):
        """Calculates the standard deviation of the distribution.

        Calculates the square root of the diagonal elements of the
        covariance matrix.  See also :obj:`covariance()` and :obj:`mean()`.

        Returns:
            The standard deviation as an n_dims array.
        """
        var = np.zeros(self.n_dims)
        for i, p in enumerate(self.particles):
            mean = np.dot(p, self.particle_weights)
            msq = np.dot(p*p, self.particle_weights)
            var[i] = msq - mean ** 2
        return np.sqrt(var)

    def bayesian_update(self, likelihood):
        """Performs a Bayesian update on the probability distribution.

        Multiplies ``particle_weights`` by the ``likelihood`` and
        renormalizes the probability
        distribution.  After the update, the distribution is tested for
        resampling depending on
        ``self.tuning_parameters['auto_resample']``.

        Args:
            likelihood: (:obj:`ndarray`):  An ``n_samples`` sized array
                describing the Bayesian likelihood of a measurement result
                calculated for each parameter combination.
         """
        self.particle_weights = self._normalized_product(self.particle_weights,
                                              likelihood)

        if self.tuning_parameters['auto_resample']:
            self.resample_test()

    def resample_test(self):
        """Tests the distribution and performs a resampling if required.

        If the effective number of particles falls below
        ``self.tuning_parameters['resample_threshold'] * n_particles``,
        performs a resampling.  Sets the ``just_resampled`` flag.
        """
        wsquared = np.nan_to_num(self.particle_weights * self.particle_weights)
        n_eff = 1 / np.sum(wsquared)
        if n_eff < 0.1 * self.n_particles:
            warnings.warn("\nParticle filter rejected > 90 % of particles. "
                          f"N_eff = {n_eff:.2f}. "
                          "Particle impoverishment may lead to errors.",
                          RuntimeWarning)
            self.resample()
            self.just_resampled = True
        # n_eff = 1 / (self.particle_weights @ self.particle_weights)
        elif n_eff / self.n_particles < \
                self.tuning_parameters['resample_threshold']:
            self.resample()
            self.just_resampled = True
        else:
            self.just_resampled = False

    def resample(self):
        """Performs a resampling of the distribution as specified by 
        and self.resampler and self.resampler_params.

        Resampling refreshes the random draws that represent the probability
        distribution.  As Bayesian updates are made, the weights of
        low-probability particles can become very small.  These particles
        consume memory and computation time, and they contribute little to
        values that are determined from the distribution.  Resampling
        abandons some low-probability particles while allowing
        high-probability particles to multiply in higher-probability regions.

        *Sample impoverishment* can occur if there are too few particles. In
        this phenomenon, a resampling step fails to sample particles from an
        important, but low-probability region, effectively removing that
        region from future consideration. The symptoms of this ``sample
        impoverishment`` phenomenon include:

            - Inconsistent results from repeated runs.  Standard deviations
              from individual final distributions will be too small to
              explain the spread of individual mean values.

            - Sudden changes in the standard deviations or other measures of
              the distribution on resampling. The resampling is not
              *supposed* to change the distribution, just refresh its
              representation.
        """

        # Call the resampler function to get a new set of particles
        # and overwrite the current particles in-place
        self.particles = self.resampler(self.particles, self.particle_weights, **self.resampler_params)
        # Re-fill the current particle weights with 1/n_particles
        self.particle_weights.fill( 1.0 / self.n_particles)

    def randdraw(self, n_draws=1):
        """Provides random parameter draws from the distribution

        Particles are selected randomly with probabilities given by
        ``self.particle_weights``.

        Args:
            n_draws (:obj:`int`): the number of draws requested.  Default
              ``1``.

        Returns:
            An ``n_dims`` x ``N_DRAWS`` :obj:`ndarray` of parameter draws.
        """
        
        return sample(self.particles,self.particle_weights,n=n_draws)

    @staticmethod
    def _normalized_product(weight_array, likelihood_array):
        """multiplies two arrays and normalizes the result.

        Functionality is added by overwriting in __init__().
        Precompiled by numba if available and if `use_jit = True`.

        Args:
            weight_array (``ndarray``): typically particle weights
            likelihood_array(``ndarray``: typically likelihoods

        Returns: a probability ``np.array`` with sum = 1.
        """
        pass

# end ParticlePDF definition

