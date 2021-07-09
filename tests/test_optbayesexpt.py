import numpy as np
from numpy.testing import assert_array_equal
from optbayesexpt import OptBayesExpt


def setup():
    pars = (np.array([0, 1, 2, 3]), np.array([1, 3, 2, 4]))
    settings = (np.array([0, 1, 2]),)
    cons = ()

    def fakefunc(sets, pars, cons):
        x, = sets
        a, b = pars
        return a + b * x

    an_obe = OptBayesExpt(fakefunc, settings, pars, cons)

    return an_obe


def test_init():
    my_obe = setup()
    pars = (np.array([0, 1, 2, 3]), np.array([1, 3, 2, 4]))
    settings = (np.array([0, 1, 2]),)

    assert_array_equal(settings, my_obe.allsettings,
                       err_msg="allsettings incorrectly initialized")
    assert_array_equal(pars, my_obe.parameters,
                       err_msg="parameters incorrectly initialized")


def test_eval_over_all_parametrs():
    my_obe = setup()
    oneset = (1, )

    assert_array_equal([[1, 4, 4, 7]], my_obe.eval_over_all_parameters(oneset),
                       err_msg="incorrect calculation")


def test_eval_over_all_settings():
    my_obe = setup()
    onepar = [1, 3]
    assert_array_equal([[1, 4, 7]], my_obe.eval_over_all_settings(onepar),
                       err_msg="incorrect calculation")


def test_likelihood():
    my_obe = setup()
    ymodel = np.array(((1, 4, 4, 7), ))
    measurement = ((1, ), (5.0, ), 1.0)
    set, ymeas, sig = measurement

    lkl = np.exp(-(ymodel - ymeas)**2 / 2)[0]
    assert_array_equal(lkl, my_obe.likelihood(ymodel, measurement),
                       err_msg="Incorrect likelihood")


def test_pdf_update():
    my_obe = setup()
    ymodel = np.array((1, 4, 4, 7))
    measurement = ((1, ), 5.0, 1.0)
    set, ymeas, sig = measurement

    lkl = np.exp(-(ymodel - ymeas)**2 / 2)
    weights = lkl/np.sum(lkl)
    my_obe.pdf_update(measurement)

    assert_array_equal(weights, my_obe.particle_weights,
                       err_msg="incorrect updated weights")

