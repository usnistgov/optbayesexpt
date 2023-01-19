"""
Tests the Bayesian inference algorithms
"""

import numpy as np
from optbayesexpt import OptBayesExpt

rng = np.random.default_rng()

n_runs = 100
n_meas_per_run = 100
n_particles = 5000
true_mean = 1.0
true_sigma = 1.0

settings = (0, )
constants = (0, )
# paramteters = (mean, sigma)


def model_function(sets, params, cons):
    """
    Minimal function.  Output = parameter
    """
    x, sig = params
    return x


class MyObe(OptBayesExpt):
    """
    A custom class that enforces parameter constraints.  Sigma > 0
    """
    def __init__(self, model_function, settings, parameters, constants):
        OptBayesExpt.__init__(self, model_function, settings,
                              parameters, constants)

    def enforce_parameter_constraints(self):
        bad_ones = np.argwhere(self.parameters[1] < 0)
        for index in bad_ones:
            self.particle_weights[index] = 0

        # renormalize
        self.particle_weights = self.particle_weights \
                               / np.sum(self.particle_weights)


def confidence95(pdf_x, pdf_w, test_x):
    """
    Determine whether test_x lies in the 95% confidence interval of the
    distribution represented by values pdf_x and weights pdf_w.
    :param pdf_x:
    :param pdf_w:
    :param test_x:
    :return:
    """
    # sort the distro by x_value
    indices = np.argsort(pdf_x)
    sorted_x = pdf_x[indices]
    sorted_w = pdf_w[indices]

    summed_w = np.cumsum(sorted_w)

    lowix = np.nonzero(summed_w > 0.025)[0][0]
    hiix = np.nonzero(summed_w < 0.975)[0][-1]
    lowlim = sorted_x[lowix]
    hilim = sorted_x[hiix]
    return lowlim <= test_x <= hilim


def do_a_run():
    x_samples = rng.uniform(-1, 5, n_particles)
    sigma_samples = rng.uniform(.2, 5, n_particles)
    parameters = (x_samples, sigma_samples)

    myobe = MyObe(model_function, settings, parameters, constants)

    meas_vals = rng.normal(true_mean, true_sigma, n_meas_per_run)
    for x in meas_vals:
        sigma_as_param = myobe.parameters[1]
        measurement = ((), x, sigma_as_param)
        myobe.pdf_update(measurement)

    passed = confidence95(myobe.parameters[0], myobe.particle_weights,
                          true_mean)

    return passed


def test_infer():
    # set up a case where the posterior distribution is calculable.
    # known x values & sigma values
    x_samples = np.linspace(-5, 5, n_particles)
    sigma_samples = np.ones(n_particles) * true_sigma
    parameters = (x_samples, sigma_samples)

    # create an OBE.
    myinferobe = MyObe(model_function, settings, parameters, constants)
    # turn off resampling
    myinferobe.tuning_parameters['resample_threshold'] = 0

    measurement = ((), true_mean, true_sigma)
    myinferobe.pdf_update(measurement)

    known_posterior = np.exp(-(true_mean - x_samples)**2 / (2*true_sigma**2))\
                             / (np.sqrt(2*np.pi) * true_sigma)
    known_posterior /= np.sum(known_posterior)
    np.testing.assert_allclose(myinferobe.particle_weights, known_posterior,
                               atol=1e-15, rtol=1e-15)


def test_experiment():
    passes = 0
    for i in np.arange(n_runs):
        if do_a_run():
            passes += 1

    assert 92 < passes < 98, f'We ran {n_runs} inference tests, expecting ' \
                             f'95 with standard deviation 2.1 to pass by ' \
                             f'yielding a result in the 95 % credible ' \
                             f'interval. In this {n_runs} tests, {passes} ' \
                             f'passed.'




