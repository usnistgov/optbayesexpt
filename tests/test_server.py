"""
Tests OBE_Server, but relies on OptBayesExpt and Socket
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from optbayesexpt import OptBayesExpt, OBE_Server, Socket
import os
from subprocess import Popen


def fakefunc(sets, pars, cons):
    x, = sets
    a, b = pars
    return a + b * x


def setup(portno=61981):
    """Starts an empty OBE_Server()

    Args:
        port: (int) port number

    Returns: OBE_Server object
    """
    pars = (np.array([0, 1, 2, 1]), np.array([1, 3, 2, 3]))
    settings = (np.array([0, 1, 0]),)
    cons = ()

    initial_args = (fakefunc, settings, pars, cons)
    # using port=port as a kwwarg raises errors - maybe a name conflict
    # using an explicit port address here avoids problems.
    an_obe = OBE_Server(initial_args=initial_args, port=61981)

    return an_obe


def test_init():
    my_obe = setup()
    pars = (np.array([0, 1, 2, 1]), np.array([1, 3, 2, 3]))
    settings = (np.array([0, 1, 0]),)
    cons = ()
    check_args = (fakefunc, settings, pars, cons)
    for check, actual in zip(check_args, my_obe.initial_args):
        assert_array_equal(check, actual,
                           err_msg='initial_args incorrectly initialized')


def test_make_obe():

    my_obe = setup()
    # different values from those used in setup()
    pars = (np.array([0, 1, 2, 3]), np.array([1, 3, 2, 4]))
    settings = (np.array([0, 1, 2]),)
    cons = ()
    class_args = (fakefunc, settings, pars, cons)

    my_obe.make_obe(OptBayesExpt, class_args)

    assert_array_equal(class_args, my_obe.initial_args,
                       err_msg='initial_args incorrectly stored')

    # spot check the obe_engine's behavior
    assert_array_equal(settings, my_obe.obe_engine.allsettings,
                       err_msg="allsettings incorrectly initialized")
    assert_array_equal(pars, my_obe.obe_engine.parameters,
                       err_msg="parameters incorrectly initialized")
    oneset = (1, )
    assert_array_equal([[1, 4, 4, 7]],
                       my_obe.obe_engine.eval_over_all_parameters(oneset),
                       err_msg="incorrect calculation")


def test_run_gets():
    # on a non-default port
    # start the script
    cwd = os.getcwd()
    server_script = os.path.join(cwd, "tests", "server_script_61982.py")
    server_pipe = Popen(['python', server_script], cwd=cwd)

    sock = Socket('client', port=61982)

    pars = (np.array([0, 1, 2, 3]), np.array([1, 3, 2, 4]))
    settings = (np.array([0, 1, 2]),)
    cons = ()

    reply = sock.tcpcmd({'command': 'getset'})
    assert_array_equal(settings, reply,
                       err_msg='settings not echoed correctly')

    reply = sock.tcpcmd({'command': 'getpar'})
    assert_array_equal(pars, reply,
                       err_msg='settings not echoed correctly')

    reply = sock.tcpcmd({'command': 'getcon'})
    assert_array_equal(cons, reply,
                       err_msg='cons not echoed correctly')

    sock.tcpcmd({'command': 'done'})


def test_run_obes():
    # on a non-default port
    # start the script
    cwd = os.getcwd()
    server_script = os.path.join(cwd, "tests", "server_script_61983.py")
    # server_pipe = Popen(['python', server_script], cwd=cwd)
    try:
        server_pipe = Popen(['python3', server_script])
    except FileNotFoundError:
        server_pipe = Popen(['python', server_script])
    sock = Socket('client', port=61983)

    # test initial weights
    weights = np.ones(4)/4.0
    reply = sock.tcpcmd({'command': 'getwgt'})
    assert_array_equal(weights, reply,
                       err_msg='weights not echoed correctly')

    # test pdf_update()
    measurement = ((1,), 5.0, 1.0)
    sets, ymeas, sig = measurement

    ymodel = np.array((1, 4, 4, 7))
    lkl = np.exp(-(ymodel - ymeas)**2 / 2)
    weights = lkl/np.sum(lkl)
    reply = sock.tcpcmd({'command': 'newdat', 'x': sets, 'y': ymeas, 's': sig})
    assert reply == "OK"

    reply = sock.tcpcmd({'command': 'getwgt'})
    assert_array_equal(weights, reply,
                       err_msg="incorrect updated weights")

    sock.tcpcmd({'command': 'done'})


def test_run_pdf():
    # on a non-default port
    # start the script
    cwd = os.getcwd()
    server_script = os.path.join(cwd, "tests", "server_script_61984.py")
    server_pipe = Popen(['python', server_script], cwd=cwd)

    sock = Socket('client', port=61984)

    right_mean = [1.5, 2.5]
    reply = sock.tcpcmd({'command': 'getmean'})
    assert_array_equal(right_mean, reply,
                       err_msg='incorrect mean returned')

    right_std = np.sqrt(np.array([1, 1]) * 5.0 / 4.0)
    reply = sock.tcpcmd({'command': 'getstd'})
    assert_array_almost_equal(right_std, reply,
                       err_msg='incorrect std deviation returned')

    right_cov = np.array([[5.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 5.0 / 3.0]])
    reply = sock.tcpcmd({'command': 'getcov'})
    assert_array_almost_equal(right_cov, reply,
                       err_msg='incorrect covariance returned')

    sock.tcpcmd({'command': 'done'})
