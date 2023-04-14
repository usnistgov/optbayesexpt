import numpy as np

rng = np.random.default_rng()

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
    num_particles = particles.shape[0]
    rng = np.random.default_rng()
    I = rng.choice(num_particles,size=n,p=weights)
    return particles[I,:]
