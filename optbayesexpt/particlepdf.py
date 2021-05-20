import numpy as np


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

    Args:
        prior (:obj:`2D array-like`): The Bayesian *prior*, which initializes the
            :obj:`ParticlePDF` distribution. Each of ``n_dims`` sub-arrays
            contains ``n_particles`` values of a single parameter, so that
            the *j*\ _th elements of the sub-arrays determine the coordinates
            of a point in parameter space. Users are encouraged to experiment
            with different ``n_particles`` sizes to assure consistent results.

    Keyword Args:
        a_param: (float) In resampling, determines the scale of random
            diffusion relative to the distribution covariance.  After
            weighted sampling, some parameter values may have been
            chosen multiple times. To make the new distribution smoother,
            the parameters are given small 'nudges', random displacements
            much smaller than the overall parameter distribution, but with
            the same shape as the overall distribution.  More precisely,
            the covariance of the nudge distribution is :code:`(1 -
            a_param ** 2)` times the covariance of the parameter distribution.
            Default ``0.98``.

        scale (:obj:`bool`): determines whether resampling includes a
            contraction of the parameter distribution toward the
            distribution mean.  The idea of this contraction is to
            compensate for the overall expansion of the distribution
            that is a by-product of random displacements.  If true,
            parameter samples (particles) move a fraction ``a_param`` of
            the distance to the distribution mean.  Default is ``True``,
            but ``False`` is recommended.

        resample_threshold (:obj:`float`): Sets a threshold for automatic
            resampling. Resampling is triggered when the effective fraction of
            particles, :math:`1 / (N\\sum_i^N w_i^2)`, is smaller than
            ``resample_threshold``.  Default ``0.5``.

        auto_resample (:obj:`bool`): Determines whether threshold testing and
            resampling are performed when ``bayesian_update()`` is called.
            Default ``True``.

    Attributes:
        n_dims (int): The number of parameters, i.e. the dimensionality of
            parameter space.  Determined from the leading dimension of
            ``prior``.

        n_particles (int): the number of parameter samples representing the
            probability distribution.  Determined from the trailing dimension
            of ``prior``.

        particles (ndarray): The ``n_particles`` samples from the
            probability distribution. Its initial value is ``prior``.

        particle_weights (ndarray): Array of n_particles probability weights
            corresponding to the particles.

        tuning_parameters (dict): A :obj:`dict` of parameters affecting the
            resampling algorithm

            - ``'a_param'`` (:obj:`float`): Initially, the value of the
              ``a_param`` keyword argument.  Default ``0.98``

            - ``'scale'`` (:obj:`bool`): Initially, the value of the
              ``scale`` keyword argument. Default ``True``

            - ``'resample_threshold'`` (:obj:`float`):  Initially,
              the value of the ``resample_threshold`` keyword argument.
              Default ``0.5``.

            - ``'auto_resample'`` (:obj:`bool`): Initially, the value of the
              ``auto_resample`` keyword argument. Default ```True``.

        just_resampled (:obj:`bool`): A flag set by the
            ``resample_test()`` function.  ``True`` if the last call
            to ``resample_test()`` resulted in resampling, else ``False``.


    Methods:
    """

    def __init__(self, prior, a_param=0.98, resample_threshold=0.5,
                 auto_resample=True, scale=True):

        self.tuning_parameters = {'a_param': a_param,
                                  'resample_threshold': resample_threshold,
                                  'auto_resample': auto_resample,
                                  'scale': scale}

        self.particles = np.asarray(prior)
        self.n_particles = self.particles.shape[-1]
        self.n_dims = self.particles.shape[0]

        self.particle_weights = np.ones(self.n_particles) / self.n_particles
        self.just_resampled = False
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
        """Calculates the covariance of the probability distribution

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
        """Calculates the standard deviation of the distribution

        Calculates the square root of the diagonal elements of the
        covariance matrix.  See also :obj:`covariance()` and :obj:`mean()`.

        Returns:
            The standard deviation as an n_dims array.
        """
        return np.sqrt(np.diag(self.covariance()))

    def bayesian_update(self, likelihood):
        """Performs a Bayesian update on the probability distribution

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
        temp = self.particle_weights * likelihood
        self.particle_weights = temp / np.sum(temp)
        if self.tuning_parameters['auto_resample']:
            self.resample_test()

    def resample_test(self):
        """Tests the distribution and performs a resampling if required.

        If the effective number of particles falls below
        ``self.tuning_parameters['resample_threshold'] * n_particles``,
        performs a resampling.  Sets the ``just_resampled`` flag.
        """
        n_eff = 1 / (self.particle_weights @ self.particle_weights)
        if n_eff / self.n_particles < \
                self.tuning_parameters['resample_threshold']:
            self.resample()
            self.just_resampled = True
        else:
            self.just_resampled = False

    def resample(self):
        """Performs a resampling of the distribution.

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
        coords = self.randdraw(self.n_particles)
        # coords is n_dims x n_particles
        origin = np.zeros(self.n_dims)

        covar = self.covariance()
        old_center = self.mean().reshape((self.n_dims, 1))
        # a_param is typically close to but less than 1
        a_param = self.tuning_parameters['a_param']
        # newcover is a small version of covar that determines the size of
        # the nudge.
        newcovar = (1 - a_param ** 2) * covar

        # multivariate normal returns n_particles x n_dims array. ".T"
        # transposes to match coords shape.
        nudged = coords + self.rng.multivariate_normal(origin, newcovar,
                                                       self.n_particles).T

        if self.tuning_parameters['scale']:
            scaled = nudged * a_param + old_center * (1 - a_param)
            self.particles = scaled
        else:
            self.particles = nudged

        self.particle_weights = np.full_like(self.particle_weights,
                                         1.0 / self.n_particles)

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

        # draws = self.rng.choice(self.particles, size=n_draws,
        # p=self.particle_weights, axis=1)
        draws = np.zeros((self.n_dims, n_draws))

        indices = self.rng.choice(self.n_particles, size=n_draws,\
                                p=self.particle_weights)

        for i, param in enumerate(self.particles):
            # for j, selected_index in enumerate(indices):
            #     draws[i,j] = param[selected_index]
            draws[i] = param[indices]

        return draws


# end ParticlePDF definition

