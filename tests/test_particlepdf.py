import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from optbayesexpt import ParticlePDF


def setup():
    prior = (np.array([0, 1, 2, 3]), np.array([1, 3, 2, 4]))
    my_pdf = ParticlePDF(prior)
    return my_pdf


#########################
#  Initialization
#########################


def test_pdf_init():
    """Tests class initialization
    """
    a_pdf = setup()
    assert_array_equal(2, a_pdf.n_dims,
                       err_msg="n_dims does not match prior dimensions")
    assert_array_equal(4, a_pdf.n_particles,
                       err_msg="n_particles does not match prior dimensions")
    assert_array_equal(np.asarray([[0, 1, 2, 3], [1, 3, 2, 4]]),
                       a_pdf.particles,
                       err_msg="prior does not match particles")
    assert_array_equal([.25, .25, .25, .25], a_pdf.particle_weights,
                       err_msg="Particle weights not initialized correctly")
    assert_array_equal(a_pdf.just_resampled, False,
                       err_msg="New ParticlePDF should have just_resampled "
                               "== False.")


def test_set_pdf():
    """Tests set_pdf() method
    """
    # create an instance
    a_pdf = setup()
    # reset the prior
    samples = np.arange(15).reshape((3, 5))

    # Testing default weights=None case
    a_pdf.set_pdf(samples)

    assert_array_equal(3, a_pdf.n_dims,
                       err_msg="n_dims does not match prior dimensions")
    assert_array_equal(5, a_pdf.n_particles,
                       err_msg="n_particles does not match prior dimensions")
    assert_array_equal(samples, a_pdf.particles,
                       err_msg="prior does not match particles")
    assert_array_equal(np.ones(5) / 5.0, a_pdf.particle_weights,
                       err_msg="Particle weights not initialized correctly")

    # testing specified weights case
    proto_weights = np.array([1, 2, 3, 4, 5])
    a_pdf.set_pdf(samples, weights=proto_weights)
    ww = np.array(proto_weights)
    rightweights = ww / np.sum(ww)
    assert_array_equal(rightweights, a_pdf.particle_weights,
                       err_msg='Incorrect particle weights')


#########################
#  Statistical Outputs
#########################


def test_mean():

    a_pdf = setup()
    meanvals = a_pdf.mean()

    assert_array_equal((a_pdf.n_dims, ), meanvals.shape,
                       err_msg="mean shape does not match n_dims")
    assert_array_equal((1.5, 2.5), meanvals,
                       err_msg="Incorrect mean values")


def test_covariance():

    a_pdf = setup()
    covar = a_pdf.covariance()
    right_answer = np.array([[5.0 / 3.0, 4.0 / 3.0],
                             [4.0 / 3.0, 5.0 / 3.0]])

    assert_array_equal((a_pdf.n_dims, a_pdf.n_dims), covar.shape,
                       err_msg="Covariance matrix has wrong shape.")
    assert_allclose(right_answer, covar,
                       err_msg="Covariance matrix has wrong values.", )


def test_std():

    a_pdf = setup()
    stdvals = a_pdf.std()
    right_answer = np.sqrt(np.array([1, 1]) * 5.0 / 4.0)

    assert_array_equal((a_pdf.n_dims, ), stdvals.shape,
                       err_msg="Standard deviation array has wrong shape.")
    assert_array_equal(right_answer, stdvals,
                       err_msg="Standard deviation array has wrong values.")


def test_bayesian_update():

    a_pdf = setup()
    # initially uniform weights
    # Turn off resampling
    a_pdf.tuning_parameters['auto_resample'] = False

    likelihood = np.array([.5, 1.5, 1.5, .5])
    right_weights = likelihood / np.sum(likelihood)
    a_pdf.bayesian_update(likelihood)

    assert_array_equal(right_weights, a_pdf.particle_weights,
                       err_msg="incorrect particle weights following update")


##########################
# Resampling
##########################


def test_resample():

    a_pdf = setup()
    a_pdf.particle_weights = np.array([0, .5, .5, 0])
    a_pdf.resample()
    print()
    print(a_pdf.particles)
    assert_array_equal((2, 4), a_pdf.particles.shape,
                       err_msg="resample returns incorrect shape")
    assert_array_equal([.25, .25, .25, .25], a_pdf.particle_weights,
                       err_msg="Particle weights not resampled correctly")


def test_resample_test():

    a_pdf = setup()
    # Effective particles = 2.94 > .5  * n_particles
    a_pdf.particle_weights = np.array([.1, .4, .4, .1])
    a_pdf.resample_test()

    assert_array_equal(False, a_pdf.just_resampled,
                       err_msg="resample triggered falsely")

    # Effective particles = 1.6 < .5 * n_particles
    a_pdf.particle_weights = np.array([0, .75, .25, 0])
    a_pdf.resample_test()
    assert_array_equal(True, a_pdf.just_resampled,
                       err_msg="resample not triggered")

