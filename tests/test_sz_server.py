
import numpy as np
import optbayesexpt as obemod
from numpy.testing import assert_array_equal

from time import sleep
from subprocess import Popen
import os

# This test focuses on testing communications with an OBE server


server_script = os.path.join(os.getcwd(), "tests/server_script.py")
server_pipe = Popen(['python', server_script])
sleep(3)


# start up a socket to communicate with server
obe = obemod.Socket(role='client', port=61982)

# settings
s1 = np.array([1, 2, 3], dtype='float')
s2 = np.array([4, 5, 6], dtype='float')

# parameters
p1 = (11, 12, 13)
p2 = [14, 15, 16]

# constants
cc = [101, 102, 103.1]


def test_addset():
    assert 'OK' == obe.tcpcmd({'command': 'addset', 'array': s1.tolist()})
    assert 'OK' == obe.tcpcmd({'command': 'addset', 'array': s2.tolist()})


def test_addpar():
    assert 'OK' == obe.tcpcmd({'command': 'addpar', 'array': p1})
    assert 'OK' == obe.tcpcmd({'command': 'addpar', 'array': p2})


def test_addcon():
    for value in cc:
        assert 'OK' == obe.tcpcmd({'command': 'addcon', 'value': value})


def test_config():
    assert 'OK' == obe.tcpcmd({'command': 'config'})


def test_getset():
    reply = obe.tcpcmd({'command': 'getset'})
    assert_array_equal(reply, [s1, s2])


def test_getpar():
    reply = obe.tcpcmd({'command': 'getpar'})
    assert_array_equal(reply, [p1, p2])


def test_getcon():
    reply = obe.tcpcmd({'command': 'getcon'})
    assert_array_equal(reply, cc)


def test_done():
    assert 'OK' == obe.tcpcmd({'command': 'done'})


if __name__ == '__main__':
    test_addset()
    test_config()
    test_getset()
    test_done()
