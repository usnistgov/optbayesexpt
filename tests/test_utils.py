import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from optbayesexpt import MeasurementSimulator, trace_sort


def setup():
    cons = (3.14, 42)
    truepars = [2., 2.]
    noise_level = 7.

    def fakefunc(sets, pars, cons):
        x, = sets
        a, b = pars
        return a + b * x

    a_sim = MeasurementSimulator(fakefunc, truepars, cons, noise_level)

    return a_sim


def test_init():
    # Tests initialization and argument list
    my_sim = setup()
    assert_array_equal([2., 2.], my_sim.params,
                       err_msg="Incorrect true_params initialization")
    assert_array_equal((3.14, 42), my_sim.cons,
                       err_msg="Incorrect cons initialization")
    assert_array_equal(7., my_sim.noise_level,
                       err_msg="Incorrect noise_level initialization")


def test_simdata_default_params():
    # test data simulation with default params=None
    my_sim = setup()
    # default truepars = [2, 2]
    settings = (np.arange(3), )
    expected_result = np.array([2, 4, 6])
    actual_result = my_sim.simdata(settings, noise_level=0.0)
    assert_array_equal(expected_result, actual_result,
                       err_msg="simdata results do not match expectations")


def test_simdata_specified_params():
    # test data simulation with default params=None
    my_sim = setup()
    params = [1., 3.]
    settings = (np.arange(3), )
    expected_result = np.array([1, 4, 7])
    actual_result = my_sim.simdata(settings, params=params, noise_level=0.0)
    assert_array_equal(expected_result, actual_result,
                       err_msg="simdata results do not match expectations")

def test_trace_sort():
    # test the trace sorting routine

    x = [1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.]
    y = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    exp_xsorted = [1, 2, 3, 4, 5, 6]
    exp_ybar = [1., 1.5, 2., 2., 2.5, 3.]
    a = .5 / np.sqrt(2)
    b = np.sqrt(2.0) / 3
    exp_std = [0, a, b, b, a, 0]
    exp_nofm = [1, 2, 3, 3, 2, 1]

    actual = trace_sort(x, y)
    act_xsorted, act_ybar, act_std, act_nofm = actual

    assert_array_equal(4, len(actual),
                       err_msg="Wrong number of items in returned tuple")
    assert_array_equal(exp_xsorted, act_xsorted,
                       err_msg="incorrect sorted setting values")
    assert_array_equal(exp_ybar, act_ybar,
                       err_msg="incorrect mean measurement values")
    assert_array_equal(exp_std, act_std,
                       err_msg="incorrect std values")
    assert_array_equal(exp_nofm, act_nofm,
                       err_msg="incorrect setting counts")