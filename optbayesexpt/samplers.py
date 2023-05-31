import numpy as np

def sample(particles, weights, n=1):
    """Provides random samples from a particle distribution.

        Particles are selected randomly with probabilities given by
        ``weights``.

        Args:
            particles (`ndarray`): The location of particles
            weights (`ndarray`): The probability weights
            n_draws (:obj:`int`): the number of samples requested.  Default
              ``1``.

        Returns:
            An ``n_dims`` x ``N_DRAWS`` :obj:`ndarray` of parameter draws.
    """
    num_particles = particles.shape[1]
    rng = np.random.default_rng()
    I = rng.choice(num_particles,size=n,p=weights)
    return particles[:,I]


def Liu_West_resampler(particles, weights, a=0.98, scale=True):
    """Resamples a particle distribution according to the Liu-West algorithm.

        Particles (``particles``) are selected randomly with probabilities given by
        ``weights``.

        Args:
            particles (`ndarray`): The location of particles

            weights (`ndarray`): The probability weights

            a_param (`float`): In resampling, determines the scale of random
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

        Returns:
            new_particles (`ndarray`): The new set of particles
    """
    rng = np.random.default_rng()
    ndim, num_particles = particles.shape
    origin = np.zeros(ndim)
    # coords is n_dims x n_particles
    coords = sample(particles, weights, n=num_particles)
    old_center = np.average(particles, axis=1, weights=weights)
    # newcovar is a small version of covar that determines the size of
    # the nudge.
    newcovar = (1-a**2)*np.cov(particles, aweights=weights, ddof=0)
    # multivariate normal returns n_particles x n_dims array. ".T"
    # transposes to match coords shape.
    nudged = coords + rng.multivariate_normal(origin, newcovar,
                                                       num_particles).T
    
    if scale:
            nudged = nudged * a
            nudged = nudged + old_center * (1 - a)

    return nudged
