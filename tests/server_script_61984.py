"""
Server script for helpme_test_server.py

listens to port 61984
"""

from optbayesexpt import OptBayesExpt, OBE_Server
import numpy as np


def fakefunc(sets, pars, cons):
    x, = sets
    a, b = pars
    return a + b * x


settings = (np.array([0, 1, 2]),)
pars = (np.array([0, 1, 2, 3]), np.array([1, 3, 2, 4]))
cons = ()

my_obe = OBE_Server(initial_args=(), port=61984)
class_args = (fakefunc, settings, pars, cons)
my_obe.make_obe(OptBayesExpt, class_args)

my_obe.run()


