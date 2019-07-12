import numpy as np
from numpy.testing import assert_array_equal

from optbayesexpt import Socket

myClient = Socket('client')
myServer = Socket('server')

def test_send_recieve_text():
    contents = 'test message'
    myClient.send(contents)
    received = myServer.receive()
    myClient.close()
    assert received == contents

def test_send_recieve_number():
    contents = 3.1415
    myClient.send(contents)
    received = myServer.receive()
    myClient.close()
    assert received == contents

def test_send_recieve_array():
    contents = [1, 2, 3]
    myClient.send(contents)
    received = myServer.receive()
    myClient.close()
    assert received == contents

"""
The tcpcmd method is tested in test_obe_server.py
"""
