""" Test experiment model class """

import numpy as np
from numpy.testing import assert_array_equal

from optbayesexpt import ExptModel

myModel = ExptModel()

s1tuple = (0, 1, 2, 3)
s2tuple = (4, 5, 6)

s1list = list(s1tuple)
s2list = list(s2tuple)

s1arr = np.arange(4, dtype='double')
s2arr = np.arange(4., 7., 1.)

p1tuple = (10, 11, 12, 13)
p2tuple = (14, 15, 16)

p1list = list(p1tuple)
p2list = list(p2tuple)

p1arr = np.arange(10., 14., 1.)
p2arr = np.arange(14., 17., 1.)

ss1 = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
ss2 = [[4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]]

pp1 = [[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13]]
pp2 = [[14, 15, 16], [14, 15, 16], [14, 15, 16], [14, 15, 16]]


def test_config_tuples():
    sets = (s1tuple, s2tuple)
    params = (p1tuple, p2tuple)
    myModel.model_config(sets, params, ())
    assert_array_equal(myModel.allsettings, [ss1, ss2])
    assert_array_equal(myModel.allparams, [pp1, pp2])

def test_config_lists():
    sets = (s1list, s2list)
    params = (p1list, p2list)
    myModel.model_config(sets, params, ())
    assert_array_equal(myModel.allsettings, [ss1, ss2])
    assert_array_equal(myModel.allparams, [pp1, pp2])

def test_config_arrays():
    sets = (s1arr, s2arr)
    params = (p1arr, p2arr)
    myModel.model_config(sets, params, ())
    assert_array_equal(myModel.allsettings, np.array([ss1, ss2], dtype='double'))
    assert_array_equal(myModel.allparams, np.array([pp1, pp2], dtype='double'))
